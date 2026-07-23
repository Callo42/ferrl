//! Safe streaming safetensors backend for rank-local tensor-parallel weights.
//!
//! Candle's buffered checkpoint loaders materialize every selected tensor on the
//! target device before the model asks for an individual weight. That defeats TP
//! memory savings. This backend reads exactly the bytes implied by each
//! `VarBuilder::get` shape: the full tensor for replicated state, an axis-0
//! range for column-parallel weights, or per-row axis-1 ranges for row-parallel
//! weights.
//!
//! Indexed checkpoints keep `weight_map` authoritative and open a shard only
//! when its declared tensor is requested. Index shard names must be one ordinary
//! relative `.safetensors` filename; directories, absolute paths, `.`/`..`, and
//! symlink shard files are rejected. Each accepted shard is opened once, its
//! complete extent is validated, and all later reads use that stable handle.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Error, Result as CandleResult, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use safetensors::tensor::{Metadata, TensorInfo};
use serde::de::{Error as _, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};

use crate::tensor_parallel::TensorParallelPlan;

const HEADER_LEN_BYTES: usize = 8;
const MAX_HEADER_LEN: usize = 100_000_000;

#[derive(Debug)]
struct OpenShard {
    path: PathBuf,
    file: Mutex<File>,
    file_len: u64,
    data_start: u64,
    metadata: Metadata,
}

#[derive(Debug, Clone)]
struct TensorSource {
    shard: Arc<OpenShard>,
    info: TensorInfo,
}

#[derive(Debug)]
enum CheckpointLayout {
    Single {
        tensors: HashMap<String, TensorSource>,
    },
    Indexed {
        dir: PathBuf,
        assignments: HashMap<String, PathBuf>,
        open_shards: Mutex<HashMap<PathBuf, Arc<OpenShard>>>,
    },
}

#[derive(Debug)]
struct ShardedSafetensorsBackend {
    layout: CheckpointLayout,
    plan: TensorParallelPlan,
}

/// Rank-invariant description of the exact, pre-opened checkpoint sources that
/// back a streaming [`VarBuilder`].
///
/// `weight_map` contains only the model-family-selected tensors. `shards` is in
/// filename order and hashes each complete retained shard through the same file
/// handle the backend subsequently uses for rank-local reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundSafetensorsIdentity {
    pub(crate) weight_map: Vec<(String, String)>,
    pub(crate) shards: Vec<(String, u64, [u8; 32])>,
}

impl ShardedSafetensorsBackend {
    #[cfg(test)]
    fn from_dir(dir: &Path, plan: TensorParallelPlan) -> CandleResult<Self> {
        ensure_supported_platform()?;
        let index_path = dir.join("model.safetensors.index.json");
        let single_path = dir.join("model.safetensors");
        let layout = if index_path.is_file() {
            let assignments = read_index(&index_path)?;
            CheckpointLayout::Indexed {
                dir: dir.to_path_buf(),
                assignments,
                open_shards: Mutex::new(HashMap::new()),
            }
        } else if single_path.is_file() {
            let shard = open_shard(&single_path)?;
            let tensors = shard
                .metadata
                .tensors()
                .into_iter()
                .map(|(name, info)| {
                    (
                        name,
                        TensorSource {
                            shard: Arc::clone(&shard),
                            info: info.clone(),
                        },
                    )
                })
                .collect::<HashMap<_, _>>();
            if tensors.is_empty() {
                return Err(msg(format!(
                    "no tensors found in checkpoint {}",
                    dir.display()
                )));
            }
            CheckpointLayout::Single { tensors }
        } else {
            return Err(msg(format!(
                "neither model.safetensors.index.json nor model.safetensors found in {}",
                dir.display()
            )));
        };
        Ok(Self { layout, plan })
    }

    fn source_for(&self, name: &str) -> CandleResult<TensorSource> {
        match &self.layout {
            CheckpointLayout::Single { tensors } => {
                tensors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| Error::CannotFindTensor {
                        path: name.to_string(),
                    })
            }
            CheckpointLayout::Indexed {
                dir,
                assignments,
                open_shards,
            } => {
                let shard_name = assignments
                    .get(name)
                    .ok_or_else(|| Error::CannotFindTensor {
                        path: name.to_string(),
                    })?;
                let shard = {
                    let mut open_shards = open_shards.lock().map_err(|_| {
                        msg("rank-local safetensors shard cache is poisoned".into())
                    })?;
                    if let Some(shard) = open_shards.get(shard_name) {
                        Arc::clone(shard)
                    } else {
                        let shard = open_shard(&dir.join(shard_name))?;
                        open_shards.insert(shard_name.clone(), Arc::clone(&shard));
                        shard
                    }
                };
                let info = shard.metadata.info(name).cloned().ok_or_else(|| {
                    msg(format!(
                        "safetensors index declares tensor {name:?} in {:?}, but that shard does not contain it",
                        shard_name.display()
                    ))
                })?;
                let source = TensorSource { shard, info };
                Ok(source)
            }
        }
    }

    fn load_expected(
        &self,
        expected: &Shape,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let source = self.source_for(name)?;
        let expected = expected.dims();
        let full = source.info.shape.as_slice();
        element_bytes(name, source.info.dtype)?;
        let bytes = if expected == full {
            read_full_tensor(&source)?
        } else {
            read_rank_local_tensor(&source, expected, self.plan, name)?
        };
        let source_dtype = DType::try_from(source.info.dtype)?;
        Tensor::from_raw_buffer(&bytes, source_dtype, expected, device)?.to_dtype(dtype)
    }
}

#[cfg(unix)]
fn ensure_supported_platform() -> CandleResult<()> {
    Ok(())
}

#[cfg(not(unix))]
fn ensure_supported_platform() -> CandleResult<()> {
    Err(msg(
        "rank-local safetensors streaming requires Unix file-identity guarantees".into(),
    ))
}

impl SimpleBackend for ShardedSafetensorsBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Tensor> {
        self.load_expected(&shape, name, dtype, device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> CandleResult<Tensor> {
        let source = self.source_for(name)?;
        self.load_expected(&Shape::from(source.info.shape.clone()), name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        match &self.layout {
            CheckpointLayout::Single { tensors } => tensors.contains_key(name),
            CheckpointLayout::Indexed { assignments, .. } => assignments.contains_key(name),
        }
    }
}

#[cfg(test)]
pub(crate) fn varbuilder_from_rank_local_safetensors(
    dir: &Path,
    dtype: DType,
    device: &Device,
    plan: TensorParallelPlan,
) -> CandleResult<VarBuilder<'static>> {
    let backend = ShardedSafetensorsBackend::from_dir(dir, plan)?;
    Ok(VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

/// Build a rank-local streaming backend and bind its identity to the same open
/// handles retained for all later tensor reads.
///
/// The selector is applied before shards are opened. This keeps a text-only
/// policy independent of vision-only shard assignments while ensuring every
/// rank hashes the same logical source set regardless of its local TP slice.
pub(crate) fn varbuilder_from_rank_local_safetensors_bound(
    dir: &Path,
    dtype: DType,
    device: &Device,
    plan: TensorParallelPlan,
    selected: impl Fn(&str) -> bool,
) -> CandleResult<(VarBuilder<'static>, BoundSafetensorsIdentity)> {
    ensure_supported_platform()?;
    let index_path = dir.join("model.safetensors.index.json");
    let single_path = dir.join("model.safetensors");
    let (layout, identity) = if index_path.is_file() {
        let assignments = read_index(&index_path)?
            .into_iter()
            .filter(|(tensor, _)| selected(tensor))
            .collect::<HashMap<_, _>>();
        if assignments.is_empty() {
            return Err(msg(format!(
                "no selected tensors found in {}",
                index_path.display()
            )));
        }
        let mut weight_map = assignments
            .iter()
            .map(|(tensor, shard)| (tensor.clone(), shard.to_string_lossy().into_owned()))
            .collect::<Vec<_>>();
        weight_map.sort();
        let shard_names = weight_map
            .iter()
            .map(|(_, shard)| shard.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let mut open_shards = HashMap::new();
        let mut shards = Vec::with_capacity(shard_names.len());
        for name in shard_names {
            let relative = PathBuf::from(&name);
            let shard = open_shard(&dir.join(&relative))?;
            let digest = hash_open_shard(&shard)?;
            shards.push((name, shard.file_len, digest));
            open_shards.insert(relative, shard);
        }
        (
            CheckpointLayout::Indexed {
                dir: dir.to_path_buf(),
                assignments,
                open_shards: Mutex::new(open_shards),
            },
            BoundSafetensorsIdentity { weight_map, shards },
        )
    } else if single_path.is_file() {
        let shard = open_shard(&single_path)?;
        let mut weight_map = shard
            .metadata
            .tensors()
            .into_iter()
            .filter(|(name, _)| selected(name))
            .map(|(name, _)| (name, "model.safetensors".to_string()))
            .collect::<Vec<_>>();
        weight_map.sort();
        if weight_map.is_empty() {
            return Err(msg(format!(
                "no selected tensors found in {}",
                single_path.display()
            )));
        }
        let tensors = weight_map
            .iter()
            .map(|(name, _)| {
                let info = shard.metadata.info(name).cloned().ok_or_else(|| {
                    msg(format!(
                        "selected tensor {name:?} disappeared from metadata"
                    ))
                })?;
                Ok((
                    name.clone(),
                    TensorSource {
                        shard: Arc::clone(&shard),
                        info,
                    },
                ))
            })
            .collect::<CandleResult<HashMap<_, _>>>()?;
        let digest = hash_open_shard(&shard)?;
        (
            CheckpointLayout::Single { tensors },
            BoundSafetensorsIdentity {
                weight_map,
                shards: vec![("model.safetensors".to_string(), shard.file_len, digest)],
            },
        )
    } else {
        return Err(msg(format!(
            "neither model.safetensors.index.json nor model.safetensors found in {}",
            dir.display()
        )));
    };
    let backend = ShardedSafetensorsBackend { layout, plan };
    Ok((
        VarBuilder::from_backend(Box::new(backend), dtype, device.clone()),
        identity,
    ))
}

fn hash_open_shard(shard: &OpenShard) -> CandleResult<[u8; 32]> {
    let mut file = shard.file.lock().map_err(|_| {
        msg(format!(
            "safetensors shard {} is poisoned",
            shard.path.display()
        ))
    })?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| msg(format!("seek {}: {error}", shard.path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut read_total = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| msg(format!("read {}: {error}", shard.path.display())))?;
        if read == 0 {
            break;
        }
        read_total = read_total
            .checked_add(read as u64)
            .ok_or_else(|| msg("safetensors shard hash length overflows u64".into()))?;
        hasher.update(&buffer[..read]);
    }
    if read_total != shard.file_len {
        return Err(msg(format!(
            "safetensors shard {} changed length while being hashed: expected {}, read {read_total}",
            shard.path.display(),
            shard.file_len
        )));
    }
    Ok(hasher.finalize().into())
}

fn read_index(index_path: &Path) -> CandleResult<HashMap<String, PathBuf>> {
    #[derive(Deserialize)]
    struct Index {
        #[serde(deserialize_with = "deserialize_unique_weight_map")]
        weight_map: HashMap<String, String>,
    }

    let bytes = std::fs::read(index_path)
        .map_err(|error| msg(format!("read {}: {error}", index_path.display())))?;
    let index: Index = serde_json::from_slice(&bytes)
        .map_err(|error| msg(format!("parse {}: {error}", index_path.display())))?;
    if index.weight_map.is_empty() {
        return Err(msg(format!("empty weight_map in {}", index_path.display())));
    }
    index
        .weight_map
        .into_iter()
        .map(|(tensor, shard)| {
            if tensor.is_empty() {
                return Err(msg(format!(
                    "empty tensor name in {} weight_map",
                    index_path.display()
                )));
            }
            Ok((tensor, validate_shard_name(index_path, &shard)?))
        })
        .collect()
}

fn deserialize_unique_weight_map<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueWeightMapVisitor;

    impl<'de> Visitor<'de> for UniqueWeightMapVisitor {
        type Value = HashMap<String, String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a weight_map with unique tensor keys")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut weight_map = HashMap::with_capacity(map.size_hint().unwrap_or(0));
            while let Some((tensor, shard)) = map.next_entry::<String, String>()? {
                match weight_map.entry(tensor) {
                    Entry::Vacant(entry) => {
                        entry.insert(shard);
                    }
                    Entry::Occupied(entry) => {
                        return Err(A::Error::custom(format!(
                            "duplicate tensor key {:?} in weight_map",
                            entry.key()
                        )));
                    }
                }
            }
            Ok(weight_map)
        }
    }

    deserializer.deserialize_map(UniqueWeightMapVisitor)
}

fn validate_shard_name(index_path: &Path, shard: &str) -> CandleResult<PathBuf> {
    let path = Path::new(shard);
    let mut components = path.components();
    let is_one_filename =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if !is_one_filename || path.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
        return Err(msg(format!(
            "invalid shard path {shard:?} in {}: expected one relative .safetensors filename",
            index_path.display()
        )));
    }
    Ok(path.to_path_buf())
}

fn open_shard(path: &Path) -> CandleResult<Arc<OpenShard>> {
    let path_metadata = std::fs::symlink_metadata(path)
        .map_err(|error| msg(format!("inspect {}: {error}", path.display())))?;
    if path_metadata.file_type().is_symlink() {
        return Err(msg(format!(
            "safetensors shard {} is a symlink; shard symlinks are not supported",
            path.display()
        )));
    }
    if !path_metadata.is_file() {
        return Err(msg(format!(
            "safetensors shard {} is not a regular file",
            path.display()
        )));
    }
    let mut file =
        File::open(path).map_err(|error| msg(format!("open {}: {error}", path.display())))?;
    let file_metadata = file
        .metadata()
        .map_err(|error| msg(format!("inspect opened {}: {error}", path.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if path_metadata.dev() != file_metadata.dev() || path_metadata.ino() != file_metadata.ino()
        {
            return Err(msg(format!(
                "safetensors shard {} changed while it was being opened",
                path.display()
            )));
        }
    }
    let file_len = file_metadata.len();
    let (data_start, metadata) = read_header(&mut file, path)?;
    let data_len = u64::try_from(metadata.data_len())
        .map_err(|_| msg(format!("{} data length does not fit u64", path.display())))?;
    let expected_len = data_start.checked_add(data_len).ok_or_else(|| {
        msg(format!(
            "{} total file length overflows u64",
            path.display()
        ))
    })?;
    if expected_len != file_len {
        return Err(msg(format!(
            "safetensors shard {} declares file length {expected_len}, actual length is {file_len}",
            path.display()
        )));
    }
    Ok(Arc::new(OpenShard {
        path: path.to_path_buf(),
        file: Mutex::new(file),
        file_len,
        data_start,
        metadata,
    }))
}

fn read_header(file: &mut File, path: &Path) -> CandleResult<(u64, Metadata)> {
    let mut len_bytes = [0_u8; HEADER_LEN_BYTES];
    file.read_exact(&mut len_bytes)
        .map_err(|error| msg(format!("read {} header length: {error}", path.display())))?;
    let header_len = usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| {
        msg(format!(
            "{} header length does not fit usize",
            path.display()
        ))
    })?;
    if header_len > MAX_HEADER_LEN {
        return Err(msg(format!(
            "{} safetensors header length {header_len} exceeds {MAX_HEADER_LEN}",
            path.display()
        )));
    }
    let mut header = vec![0_u8; header_len];
    file.read_exact(&mut header)
        .map_err(|error| msg(format!("read {} header: {error}", path.display())))?;
    let metadata: Metadata = serde_json::from_slice(&header)
        .map_err(|error| msg(format!("parse {} header: {error}", path.display())))?;
    let data_start = u64::try_from(HEADER_LEN_BYTES)
        .ok()
        .and_then(|prefix| prefix.checked_add(u64::try_from(header_len).ok()?))
        .ok_or_else(|| msg(format!("{} header extent overflows u64", path.display())))?;
    Ok((data_start, metadata))
}

fn read_full_tensor(source: &TensorSource) -> CandleResult<Vec<u8>> {
    let (start, end) = source.info.data_offsets;
    let len = end.checked_sub(start).ok_or_else(|| {
        msg(format!(
            "tensor offset underflow in {}",
            source.shard.path.display()
        ))
    })?;
    read_tensor_range(source, 0, len)
}

fn read_rank_local_tensor(
    source: &TensorSource,
    expected: &[usize],
    plan: TensorParallelPlan,
    name: &str,
) -> CandleResult<Vec<u8>> {
    if !plan.is_sharded() {
        return Err(shape_error(name, expected, &source.info.shape));
    }
    let full = source.info.shape.as_slice();
    if full.len() != 2 || expected.len() != 2 {
        return Err(msg(format!(
            "rank-local tensor {name:?} must be a matrix, checkpoint shape {full:?}, requested {expected:?}"
        )));
    }
    let world = plan.world_size();
    let axis = match (
        expected[0].checked_mul(world) == Some(full[0]) && expected[1] == full[1],
        expected[0] == full[0] && expected[1].checked_mul(world) == Some(full[1]),
    ) {
        (true, false) => 0,
        (false, true) => 1,
        _ => return Err(shape_error(name, expected, full)),
    };
    let element_bytes = element_bytes(name, source.info.dtype)?;
    if axis == 0 {
        let row_bytes = full[1]
            .checked_mul(element_bytes)
            .ok_or_else(|| msg(format!("tensor {name:?} row byte size overflow")))?;
        let byte_len = expected[0]
            .checked_mul(row_bytes)
            .ok_or_else(|| msg(format!("tensor {name:?} shard byte size overflow")))?;
        let rank_rows = plan
            .rank()
            .checked_mul(expected[0])
            .ok_or_else(|| msg(format!("tensor {name:?} row offset overflow")))?;
        let start = rank_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| msg(format!("tensor {name:?} byte offset overflow")))?;
        return read_tensor_range(source, start, byte_len);
    }

    let row_bytes = expected[1]
        .checked_mul(element_bytes)
        .ok_or_else(|| msg(format!("tensor {name:?} row-shard byte size overflow")))?;
    let output_len = expected[0]
        .checked_mul(row_bytes)
        .ok_or_else(|| msg(format!("tensor {name:?} row-shard allocation overflow")))?;
    let mut bytes = vec![0_u8; output_len];
    let full_row_bytes = full[1]
        .checked_mul(element_bytes)
        .ok_or_else(|| msg(format!("tensor {name:?} full row byte size overflow")))?;
    let column_offset = plan
        .rank()
        .checked_mul(row_bytes)
        .ok_or_else(|| msg(format!("tensor {name:?} column offset overflow")))?;
    let mut file = source.shard.file.lock().map_err(|_| {
        msg(format!(
            "safetensors shard {} is poisoned",
            source.shard.path.display()
        ))
    })?;
    for row in 0..expected[0] {
        let row_offset = row
            .checked_mul(full_row_bytes)
            .and_then(|offset| offset.checked_add(column_offset))
            .ok_or_else(|| msg(format!("tensor {name:?} row offset overflow")))?;
        let output_start = row
            .checked_mul(row_bytes)
            .ok_or_else(|| msg(format!("tensor {name:?} output offset overflow")))?;
        read_tensor_range_into(
            source,
            &mut file,
            row_offset,
            &mut bytes[output_start..output_start + row_bytes],
        )?;
    }
    Ok(bytes)
}

fn element_bytes(name: &str, dtype: safetensors::Dtype) -> CandleResult<usize> {
    let bits = dtype.bitsize();
    if bits == 0 || !bits.is_multiple_of(8) {
        return Err(msg(format!(
            "tensor {name:?} uses unsupported non-byte-aligned dtype {dtype:?}"
        )));
    }
    Ok(bits / 8)
}

fn read_tensor_range(
    source: &TensorSource,
    tensor_relative_start: usize,
    len: usize,
) -> CandleResult<Vec<u8>> {
    let mut bytes = vec![0_u8; len];
    let mut file = source.shard.file.lock().map_err(|_| {
        msg(format!(
            "safetensors shard {} is poisoned",
            source.shard.path.display()
        ))
    })?;
    read_tensor_range_into(source, &mut file, tensor_relative_start, &mut bytes)?;
    Ok(bytes)
}

fn read_tensor_range_into(
    source: &TensorSource,
    file: &mut File,
    tensor_relative_start: usize,
    output: &mut [u8],
) -> CandleResult<()> {
    let (tensor_start, tensor_end) = source.info.data_offsets;
    let tensor_len = tensor_end.checked_sub(tensor_start).ok_or_else(|| {
        msg(format!(
            "tensor offset underflow in {}",
            source.shard.path.display()
        ))
    })?;
    let range_end = tensor_relative_start
        .checked_add(output.len())
        .ok_or_else(|| msg("tensor read range overflows usize".into()))?;
    if range_end > tensor_len {
        return Err(msg(format!(
            "tensor read range {tensor_relative_start}..{range_end} exceeds tensor length {tensor_len} in {}",
            source.shard.path.display()
        )));
    }
    let relative = tensor_start
        .checked_add(tensor_relative_start)
        .ok_or_else(|| msg("tensor file-relative offset overflows usize".into()))?;
    let relative = u64::try_from(relative)
        .map_err(|_| msg("tensor file-relative offset does not fit u64".into()))?;
    let absolute = source
        .shard
        .data_start
        .checked_add(relative)
        .ok_or_else(|| msg("tensor absolute offset overflows u64".into()))?;
    let absolute_end = absolute
        .checked_add(
            u64::try_from(output.len())
                .map_err(|_| msg("tensor read length does not fit u64".into()))?,
        )
        .ok_or_else(|| msg("tensor absolute end overflows u64".into()))?;
    if absolute_end > source.shard.file_len {
        return Err(msg(format!(
            "tensor read ending at {absolute_end} exceeds file length {} in {}",
            source.shard.file_len,
            source.shard.path.display()
        )));
    }
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| msg(format!("seek {}: {error}", source.shard.path.display())))?;
    file.read_exact(output)
        .map_err(|error| msg(format!("read {}: {error}", source.shard.path.display())))
}

fn shape_error(name: &str, requested: &[usize], checkpoint: &[usize]) -> Error {
    msg(format!(
        "rank-local shape mismatch for {name:?}: checkpoint {checkpoint:?}, requested {requested:?}"
    ))
}

fn msg(message: String) -> Error {
    Error::Msg(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::safetensors;
    use std::io::Write as _;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "ferrl-sharded-safetensors-{label}-{}-{}",
                std::process::id(),
                std::thread::current().name().unwrap_or("unnamed")
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn matrix(rows: usize, cols: usize) -> Tensor {
        let values: Vec<f32> = (0..rows * cols).map(|value| value as f32).collect();
        Tensor::from_vec(values, (rows, cols), &Device::Cpu).unwrap()
    }

    fn write_index(dir: &Path, weight_map: &serde_json::Value) {
        let index = serde_json::json!({ "weight_map": weight_map });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
    }

    fn write_raw_safetensors(path: &Path, header: &serde_json::Value, data: &[u8]) {
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(data).unwrap();
    }

    #[test]
    fn tensor_parallel_backend_reads_column_row_and_replicated_shapes_for_each_rank() {
        let dir = TestDir::new("axes");
        let column = matrix(8, 4);
        let row = matrix(4, 8);
        let replicated = matrix(2, 3);
        let tensors = HashMap::from([
            ("column.weight", column.clone()),
            ("row.weight", row.clone()),
            ("replicated.weight", replicated.clone()),
        ]);
        safetensors::save(&tensors, dir.0.join("model.safetensors")).unwrap();

        for rank in 0..2 {
            let plan = TensorParallelPlan::new(rank, 2).unwrap();
            let vb = varbuilder_from_rank_local_safetensors(&dir.0, DType::F32, &Device::Cpu, plan)
                .unwrap();
            let got_column = vb.get((4, 4), "column.weight").unwrap();
            let got_row = vb.get((4, 4), "row.weight").unwrap();
            let got_replicated = vb.get((2, 3), "replicated.weight").unwrap();

            let want_column = column.narrow(0, rank * 4, 4).unwrap();
            let want_row = row.narrow(1, rank * 4, 4).unwrap();
            assert_eq!(
                got_column.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                want_column.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
            assert_eq!(
                got_row.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                want_row.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
            assert_eq!(
                got_replicated
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                replicated.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
    }

    #[test]
    fn tensor_parallel_backend_reads_indexed_checkpoint_shards() {
        let dir = TestDir::new("index");
        let column = matrix(8, 4);
        let row = matrix(4, 8);
        safetensors::save(
            &HashMap::from([("column.weight", column.clone())]),
            dir.0.join("model-00001-of-00002.safetensors"),
        )
        .unwrap();
        safetensors::save(
            &HashMap::from([("row.weight", row.clone())]),
            dir.0.join("model-00002-of-00002.safetensors"),
        )
        .unwrap();
        let index = serde_json::json!({
            "weight_map": {
                "column.weight": "model-00001-of-00002.safetensors",
                "row.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            dir.0.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();

        let plan = TensorParallelPlan::new(1, 2).unwrap();
        let vb =
            varbuilder_from_rank_local_safetensors(&dir.0, DType::F32, &Device::Cpu, plan).unwrap();
        let got_column = vb.get((4, 4), "column.weight").unwrap();
        let got_row = vb.get((4, 4), "row.weight").unwrap();
        assert_eq!(
            got_column.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            column
                .narrow(0, 4, 4)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert_eq!(
            got_row.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            row.narrow(1, 4, 4)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn tensor_parallel_backend_rejects_shapes_that_are_not_one_rank_local_axis() {
        let dir = TestDir::new("shape");
        let weight = matrix(8, 8);
        safetensors::save(
            &HashMap::from([("weight", weight)]),
            dir.0.join("model.safetensors"),
        )
        .unwrap();
        let vb = varbuilder_from_rank_local_safetensors(
            &dir.0,
            DType::F32,
            &Device::Cpu,
            TensorParallelPlan::new(0, 2).unwrap(),
        )
        .unwrap();

        let error = vb.get((4, 4), "weight").unwrap_err().to_string();
        assert!(error.contains("rank-local shape mismatch"), "{error}");
    }

    #[test]
    fn indexed_backend_enforces_tensor_to_shard_assignments_and_ignores_undeclared_tensors() {
        let dir = TestDir::new("authoritative-index");
        let declared = matrix(2, 3);
        safetensors::save(
            &HashMap::from([("declared.weight", declared)]),
            dir.0.join("physical.safetensors"),
        )
        .unwrap();
        safetensors::save(
            &HashMap::from([
                ("other.weight", matrix(2, 3)),
                ("undeclared.weight", matrix(1, 2)),
            ]),
            dir.0.join("wrong.safetensors"),
        )
        .unwrap();
        write_index(
            &dir.0,
            &serde_json::json!({
                "declared.weight": "wrong.safetensors"
            }),
        );

        let backend =
            ShardedSafetensorsBackend::from_dir(&dir.0, TensorParallelPlan::new(0, 2).unwrap())
                .unwrap();
        assert!(!backend.contains_tensor("undeclared.weight"));
        assert!(matches!(
            backend.source_for("undeclared.weight"),
            Err(Error::CannotFindTensor { .. })
        ));
        let error = backend
            .load_expected(
                &Shape::from((2, 3)),
                "declared.weight",
                DType::F32,
                &Device::Cpu,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("index declares tensor"), "{error}");
        assert!(error.contains("wrong.safetensors"), "{error}");
    }

    #[test]
    fn indexed_backend_rejects_duplicate_tensor_assignments() {
        let dir = TestDir::new("duplicate-index-key");
        std::fs::write(
            dir.0.join("model.safetensors.index.json"),
            br#"{"weight_map":{"weight":"one.safetensors","weight":"two.safetensors"}}"#,
        )
        .unwrap();

        let error = read_index(&dir.0.join("model.safetensors.index.json"))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("duplicate tensor key \"weight\" in weight_map"),
            "{error}"
        );
    }

    #[test]
    fn indexed_backend_rejects_escaping_and_non_shard_paths() {
        for (label, shard) in [
            ("absolute", "/tmp/outside.safetensors"),
            ("parent", "../outside.safetensors"),
            ("nested", "nested/model.safetensors"),
            ("extension", "model.bin"),
        ] {
            let dir = TestDir::new(label);
            write_index(&dir.0, &serde_json::json!({ "weight": shard }));
            let error =
                ShardedSafetensorsBackend::from_dir(&dir.0, TensorParallelPlan::new(0, 2).unwrap())
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("invalid shard path"), "{label}: {error}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn indexed_backend_rejects_symlink_shards_when_requested() {
        use std::os::unix::fs::symlink;

        let dir = TestDir::new("symlink");
        safetensors::save(
            &HashMap::from([("weight", matrix(2, 3))]),
            dir.0.join("target.safetensors"),
        )
        .unwrap();
        symlink("target.safetensors", dir.0.join("link.safetensors")).unwrap();
        write_index(&dir.0, &serde_json::json!({ "weight": "link.safetensors" }));
        let backend =
            ShardedSafetensorsBackend::from_dir(&dir.0, TensorParallelPlan::new(0, 2).unwrap())
                .unwrap();

        let error = backend.source_for("weight").unwrap_err().to_string();
        assert!(error.contains("is a symlink"), "{error}");
    }

    #[test]
    fn indexed_backend_opens_only_the_requested_text_shard() {
        let dir = TestDir::new("text-only");
        let text = matrix(2, 3);
        safetensors::save(
            &HashMap::from([("model.language_model.weight", text.clone())]),
            dir.0.join("text.safetensors"),
        )
        .unwrap();
        write_index(
            &dir.0,
            &serde_json::json!({
                "model.language_model.weight": "text.safetensors",
                "model.vision.weight": "missing-vision.safetensors"
            }),
        );
        let vb = varbuilder_from_rank_local_safetensors(
            &dir.0,
            DType::F32,
            &Device::Cpu,
            TensorParallelPlan::new(0, 2).unwrap(),
        )
        .unwrap();

        let got = vb.get((2, 3), "model.language_model.weight").unwrap();
        assert_eq!(
            got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            text.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn backend_rejects_truncated_extent_and_out_of_tensor_reads() {
        let truncated = TestDir::new("truncated");
        let path = truncated.0.join("model.safetensors");
        safetensors::save(&HashMap::from([("weight", matrix(2, 3))]), &path).unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 1)
            .unwrap();
        let error = ShardedSafetensorsBackend::from_dir(
            &truncated.0,
            TensorParallelPlan::new(0, 2).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("declares file length"), "{error}");

        let bounded = TestDir::new("bounded-read");
        safetensors::save(
            &HashMap::from([("weight", matrix(2, 3))]),
            bounded.0.join("model.safetensors"),
        )
        .unwrap();
        let backend =
            ShardedSafetensorsBackend::from_dir(&bounded.0, TensorParallelPlan::new(0, 2).unwrap())
                .unwrap();
        let source = backend.source_for("weight").unwrap();
        let error = read_tensor_range(&source, 23, 2).unwrap_err().to_string();
        assert!(error.contains("exceeds tensor length 24"), "{error}");
    }

    #[test]
    fn backend_rejects_sub_byte_dtypes_explicitly() {
        let dir = TestDir::new("sub-byte");
        write_raw_safetensors(
            &dir.0.join("model.safetensors"),
            &serde_json::json!({
                "weight": {
                    "dtype": "F4",
                    "shape": [2],
                    "data_offsets": [0, 1]
                }
            }),
            &[0],
        );
        let backend =
            ShardedSafetensorsBackend::from_dir(&dir.0, TensorParallelPlan::new(0, 2).unwrap())
                .unwrap();

        let error = backend
            .load_expected(&Shape::from((2,)), "weight", DType::F32, &Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("unsupported non-byte-aligned dtype F4"),
            "{error}"
        );
    }

    #[test]
    fn backend_reads_from_the_validated_file_after_path_replacement() {
        let dir = TestDir::new("replacement");
        let path = dir.0.join("model.safetensors");
        let original = matrix(2, 3);
        safetensors::save(&HashMap::from([("weight", original.clone())]), &path).unwrap();
        let backend =
            ShardedSafetensorsBackend::from_dir(&dir.0, TensorParallelPlan::new(0, 2).unwrap())
                .unwrap();
        std::fs::rename(&path, dir.0.join("original.safetensors")).unwrap();
        let replacement = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
        safetensors::save(&HashMap::from([("weight", replacement)]), &path).unwrap();

        let got = backend
            .load_expected(&Shape::from((2, 3)), "weight", DType::F32, &Device::Cpu)
            .unwrap();
        assert_eq!(
            got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            original.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn bound_backend_identity_and_reads_survive_replacement_between_hash_and_load() {
        let dir = TestDir::new("bound-replacement");
        let path = dir.0.join("model.safetensors");
        let original = matrix(2, 3);
        safetensors::save(&HashMap::from([("weight", original.clone())]), &path).unwrap();
        let (vb, original_identity) = varbuilder_from_rank_local_safetensors_bound(
            &dir.0,
            DType::F32,
            &Device::Cpu,
            TensorParallelPlan::new(0, 2).unwrap(),
            |_| true,
        )
        .unwrap();

        std::fs::rename(&path, dir.0.join("original.safetensors")).unwrap();
        let replacement = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
        safetensors::save(&HashMap::from([("weight", replacement)]), &path).unwrap();
        let (_replacement_vb, replacement_identity) = varbuilder_from_rank_local_safetensors_bound(
            &dir.0,
            DType::F32,
            &Device::Cpu,
            TensorParallelPlan::new(0, 2).unwrap(),
            |_| true,
        )
        .unwrap();
        assert_ne!(original_identity, replacement_identity);

        let got = vb.get((2, 3), "weight").unwrap();
        assert_eq!(
            got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            original.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "the bound backend must load through the exact handle it hashed"
        );
    }
}
