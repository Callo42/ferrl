//! Safe streaming safetensors backend for rank-local tensor-parallel weights.
//!
//! Candle's buffered checkpoint loaders materialize every selected tensor on the
//! target device before the model asks for an individual weight. That defeats TP
//! memory savings. This backend parses only safetensors headers up front and reads
//! exactly the bytes implied by each `VarBuilder::get` shape: the full tensor for
//! replicated state, an axis-0 range for column-parallel weights, or per-row
//! axis-1 ranges for row-parallel weights.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Error, Result as CandleResult, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use safetensors::tensor::{Metadata, TensorInfo};
use serde::Deserialize;

use crate::tensor_parallel::TensorParallelPlan;

const HEADER_LEN_BYTES: usize = 8;
const MAX_HEADER_LEN: usize = 100_000_000;

#[derive(Debug, Clone)]
struct TensorSource {
    path: PathBuf,
    data_start: u64,
    info: TensorInfo,
}

#[derive(Debug)]
struct ShardedSafetensorsBackend {
    tensors: HashMap<String, TensorSource>,
    plan: TensorParallelPlan,
}

impl ShardedSafetensorsBackend {
    fn from_dir(dir: &Path, plan: TensorParallelPlan) -> CandleResult<Self> {
        let files = checkpoint_files(dir)?;
        let mut tensors = HashMap::new();
        for path in files {
            let (data_start, metadata) = read_header(&path)?;
            for (name, info) in metadata.tensors() {
                let source = TensorSource {
                    path: path.clone(),
                    data_start,
                    info: info.clone(),
                };
                if tensors.insert(name.clone(), source).is_some() {
                    return Err(msg(format!(
                        "tensor {name:?} appears in more than one safetensors shard"
                    )));
                }
            }
        }
        if tensors.is_empty() {
            return Err(msg(format!(
                "no tensors found in checkpoint {}",
                dir.display()
            )));
        }
        Ok(Self { tensors, plan })
    }

    fn load_expected(
        &self,
        expected: &Shape,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let source = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::CannotFindTensor {
                path: name.to_string(),
            })?;
        let expected = expected.dims();
        let full = source.info.shape.as_slice();
        let bytes = if expected == full {
            read_full_tensor(source)?
        } else {
            read_rank_local_tensor(source, expected, self.plan, name)?
        };
        let source_dtype = DType::try_from(source.info.dtype)?;
        Tensor::from_raw_buffer(&bytes, source_dtype, expected, device)?.to_dtype(dtype)
    }
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
        let source = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::CannotFindTensor {
                path: name.to_string(),
            })?;
        self.load_expected(&Shape::from(source.info.shape.clone()), name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

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

fn checkpoint_files(dir: &Path) -> CandleResult<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    let single_path = dir.join("model.safetensors");
    if index_path.is_file() {
        #[derive(Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }

        let bytes = std::fs::read(&index_path)
            .map_err(|error| msg(format!("read {}: {error}", index_path.display())))?;
        let index: Index = serde_json::from_slice(&bytes)
            .map_err(|error| msg(format!("parse {}: {error}", index_path.display())))?;
        let mut names: Vec<_> = index
            .weight_map
            .into_values()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        if names.is_empty() {
            return Err(msg(format!("empty weight_map in {}", index_path.display())));
        }
        Ok(names.into_iter().map(|name| dir.join(name)).collect())
    } else if single_path.is_file() {
        Ok(vec![single_path])
    } else {
        Err(msg(format!(
            "neither model.safetensors.index.json nor model.safetensors found in {}",
            dir.display()
        )))
    }
}

fn read_header(path: &Path) -> CandleResult<(u64, Metadata)> {
    let mut file =
        File::open(path).map_err(|error| msg(format!("open {}: {error}", path.display())))?;
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
    Ok(((HEADER_LEN_BYTES + header_len) as u64, metadata))
}

fn read_full_tensor(source: &TensorSource) -> CandleResult<Vec<u8>> {
    let (start, end) = source.info.data_offsets;
    read_range(source, start, end - start)
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
    let element_bytes = source
        .info
        .dtype
        .bitsize()
        .checked_div(8)
        .ok_or_else(|| msg(format!("tensor {name:?} dtype is not byte-aligned")))?;
    if axis == 0 {
        let row_bytes = full[1]
            .checked_mul(element_bytes)
            .ok_or_else(|| msg(format!("tensor {name:?} row byte size overflow")))?;
        let byte_len = expected[0]
            .checked_mul(row_bytes)
            .ok_or_else(|| msg(format!("tensor {name:?} shard byte size overflow")))?;
        let start = source.info.data_offsets.0 + plan.rank() * expected[0] * row_bytes;
        return read_range(source, start, byte_len);
    }

    let row_bytes = expected[1]
        .checked_mul(element_bytes)
        .ok_or_else(|| msg(format!("tensor {name:?} row-shard byte size overflow")))?;
    let mut bytes = vec![0_u8; expected[0] * row_bytes];
    let full_row_bytes = full[1] * element_bytes;
    let column_offset = plan.rank() * row_bytes;
    let mut file = File::open(&source.path)
        .map_err(|error| msg(format!("open {}: {error}", source.path.display())))?;
    for row in 0..expected[0] {
        let relative = source.info.data_offsets.0 + row * full_row_bytes + column_offset;
        file.seek(SeekFrom::Start(source.data_start + relative as u64))
            .map_err(|error| msg(format!("seek {}: {error}", source.path.display())))?;
        file.read_exact(&mut bytes[row * row_bytes..(row + 1) * row_bytes])
            .map_err(|error| msg(format!("read {}: {error}", source.path.display())))?;
    }
    Ok(bytes)
}

fn read_range(source: &TensorSource, relative_start: usize, len: usize) -> CandleResult<Vec<u8>> {
    let mut file = File::open(&source.path)
        .map_err(|error| msg(format!("open {}: {error}", source.path.display())))?;
    file.seek(SeekFrom::Start(source.data_start + relative_start as u64))
        .map_err(|error| msg(format!("seek {}: {error}", source.path.display())))?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes)
        .map_err(|error| msg(format!("read {}: {error}", source.path.display())))?;
    Ok(bytes)
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
}
