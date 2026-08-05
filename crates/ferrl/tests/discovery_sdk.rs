//! Downstream compatibility fixture for the stable discovery facade.

use ferrl::discovery::{
    Candidate, CandidateVerification, Discovery, DiscoveryConfig, DiscoveryError, DiscoveryOutcome,
    DiscoveryTask, FinalEvidence, GenerationEnd, MetricContract, MetricDirection, MetricReport,
    ModelSelection, RewardError, RewardFn, Sample, TaskIdentity, TaskVerificationError,
};

struct ContainsReward;

impl RewardFn for ContainsReward {
    type Target = String;

    fn reward(&self, sample: &Sample<Self::Target>, completion: &str) -> Result<f32, RewardError> {
        Ok(if completion.contains(&sample.target) {
            1.0
        } else {
            0.0
        })
    }
}

struct DownstreamTask {
    identity: TaskIdentity,
    train: Vec<Sample<String>>,
    held_out: Vec<Sample<String>>,
    reward: ContainsReward,
}

impl DiscoveryTask for DownstreamTask {
    type Target = String;
    type SearchReward = ContainsReward;
    type Artifact = String;
    type VerificationEvidence = String;

    fn identity(&self) -> &TaskIdentity {
        &self.identity
    }

    fn training_samples(&self) -> &[Sample<Self::Target>] {
        &self.train
    }

    fn held_out_samples(&self) -> &[Sample<Self::Target>] {
        &self.held_out
    }

    fn search_reward(&self) -> &Self::SearchReward {
        &self.reward
    }

    fn metric_contract(&self) -> MetricContract {
        MetricContract::new(
            "throughput",
            "items/s",
            MetricDirection::HigherIsBetter,
            100.0,
            2.0,
        )
    }

    fn verify_candidate(
        &self,
        candidate: Candidate<'_>,
    ) -> Result<
        CandidateVerification<Self::Artifact, Self::VerificationEvidence>,
        TaskVerificationError,
    > {
        let metric = MetricReport::new(
            "throughput",
            "items/s",
            MetricDirection::HigherIsBetter,
            100.0,
            103.0,
            2.0,
        );
        Ok(CandidateVerification::measured(FinalEvidence::new(
            candidate.completion().to_owned(),
            "downstream verifier evidence".to_owned(),
            true,
            metric,
        )))
    }
}

fn consume_outcome(outcome: DiscoveryOutcome) {
    match outcome {
        DiscoveryOutcome::Verified(artifact) => {
            let _ = (
                artifact.output(),
                artifact.manifest_path(),
                artifact.payload_path(),
                artifact.candidate_path(),
                artifact.verification_evidence_path(),
                artifact.task_identity(),
                artifact.model_identity(),
                artifact.candidate_sha256(),
                artifact.metric(),
            );
        }
        DiscoveryOutcome::NoWin(report) => {
            let _ = (
                report.run_dir(),
                report.held_out(),
                report.candidates_checked(),
                report.reason(),
                report.detail(),
            );
        }
        DiscoveryOutcome::Preempted(report) => {
            let _ = (
                report.run_dir(),
                report.completed_steps(),
                report.checkpoint_path(),
            );
        }
    }
}

#[test]
fn downstream_contract_fixes_construction_extension_and_outcome_names() {
    let task = DownstreamTask {
        identity: TaskIdentity::new("downstream.contains", 1).unwrap(),
        train: vec![Sample::new("emit alpha", "alpha".to_owned())],
        held_out: vec![Sample::new("emit beta", "beta".to_owned())],
        reward: ContainsReward,
    };
    let model = ModelSelection::cpu("checkpoint").generation_end(GenerationEnd::Disabled);
    let config = DiscoveryConfig::builder("runs", "artifacts/winner")
        .steps(4)
        .group_size(4)
        .max_new_tokens(32)
        .eval_group_size(2)
        .temperature(0.8)
        .learning_rate(1e-4)
        .seed(7)
        .build()
        .unwrap();
    let discovery = Discovery::new(task, model, config);
    let _ = discovery;
    let _run_signature = Discovery::<DownstreamTask>::run
        as fn(Discovery<DownstreamTask>) -> Result<DiscoveryOutcome, DiscoveryError>;
    let _handler: fn(DiscoveryOutcome) = consume_outcome;
}
