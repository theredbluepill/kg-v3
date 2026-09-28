//! Exact RQ60/RQ61 crop-modified complete-matching candidate enumeration.
//!
//! This module intentionally owns only the graph kernel. R04 still proposes the
//! units, jobs, and scored edges in Python, and Python still materializes the
//! selected matching into low-level actions. The implementation mirrors the
//! rectangular Hungarian and candidate/tie semantics in
//! `rq61_joint_family_actor._joint_rows`.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

pub const FORCE_RESIDUAL: f64 = 1_000_000_000.0;
const MISSING_COST: f64 = 1.0e15;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct EdgeRecord {
    pub index: usize,
    pub base_score: f64,
    pub unit: usize,
    pub job: usize,
    #[serde(default)]
    pub extra_walk: i64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct JointRequest {
    pub n_units: usize,
    pub n_jobs: usize,
    pub edges: Vec<EdgeRecord>,
    pub baseline_matching: Vec<[usize; 2]>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct JointRow {
    pub forced_edge_indices: Vec<usize>,
    pub forced_pairs: Vec<[usize; 2]>,
    pub forced_base_scores: Vec<f64>,
    pub matching: Vec<[usize; 2]>,
    pub heuristic_score: f64,
    pub rank: usize,
    pub matching_size: usize,
    pub changed_count: usize,
    pub heuristic_delta: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct JointBenchRequest {
    pub requests: Vec<JointRequest>,
    pub repetitions: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct JointBenchResult {
    pub graph_calls: usize,
    pub elapsed_ns: u128,
    pub checksum: usize,
}

type Pair = (usize, usize);
type Matching = Vec<Pair>;

fn matching_key(rows: impl IntoIterator<Item = Pair>) -> Matching {
    let mut result: Matching = rows.into_iter().collect();
    result.sort_unstable();
    result
}

fn validate(request: &JointRequest) -> Result<(), String> {
    for (position, edge) in request.edges.iter().enumerate() {
        if edge.index != position {
            return Err(format!(
                "edge index {} at position {position}; Python residual semantics require contiguous positional indices",
                edge.index
            ));
        }
        if !edge.base_score.is_finite() {
            return Err(format!("edge {} has non-finite base_score", edge.index));
        }
        if edge.unit >= request.n_units {
            return Err(format!(
                "edge {} unit {} outside n_units {}",
                edge.index, edge.unit, request.n_units
            ));
        }
        if edge.job >= request.n_jobs {
            return Err(format!(
                "edge {} job {} outside n_jobs {}",
                edge.index, edge.job, request.n_jobs
            ));
        }
    }
    for pair in &request.baseline_matching {
        if pair[0] >= request.n_units || pair[1] >= request.n_jobs {
            return Err(format!(
                "baseline pair ({}, {}) is outside {}x{}",
                pair[0], pair[1], request.n_units, request.n_jobs
            ));
        }
    }
    Ok(())
}

/// R04's rectangular maximum-score Hungarian solve with one zero-score idle
/// column per unit and unavailable real edges assigned a cost of `1e15`.
fn decode_matching(request: &JointRequest, forced_indices: &[usize]) -> Result<Matching, String> {
    if request.n_units == 0 || request.edges.is_empty() {
        return Ok(Vec::new());
    }
    let forced: BTreeSet<usize> = forced_indices.iter().copied().collect();
    if let Some(index) = forced.iter().find(|index| **index >= request.edges.len()) {
        return Err(format!(
            "forced edge index {index} is outside the edge list"
        ));
    }

    // Python's dict keeps the first pair unless a later score is strictly
    // greater. Preserve that rule, including duplicate unit/job edges.
    let mut best: BTreeMap<Pair, (f64, usize)> = BTreeMap::new();
    for edge in &request.edges {
        let score = edge.base_score
            + if forced.contains(&edge.index) {
                FORCE_RESIDUAL
            } else {
                0.0
            };
        let key = (edge.unit, edge.job);
        match best.get(&key) {
            None => {
                best.insert(key, (score, edge.index));
            }
            Some((old, _)) if score > *old => {
                best.insert(key, (score, edge.index));
            }
            _ => {}
        }
    }

    let n = request.n_units;
    let m = request.n_jobs + request.n_units;
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut matched_row = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    let cost = |row1: usize, col1: usize| -> f64 {
        let col = col1 - 1;
        if col >= request.n_jobs {
            0.0
        } else {
            best.get(&(row1 - 1, col))
                .map_or(MISSING_COST, |(score, _)| -*score)
        }
    };

    for row1 in 1..=n {
        matched_row[0] = row1;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        let mut col0 = 0usize;
        loop {
            used[col0] = true;
            let row0 = matched_row[col0];
            let mut delta = f64::INFINITY;
            let mut col1 = 0usize;
            for col in 1..=m {
                if used[col] {
                    continue;
                }
                let cur = cost(row0, col) - u[row0] - v[col];
                if cur < minv[col] {
                    minv[col] = cur;
                    way[col] = col0;
                }
                // Strict comparison is the Python/R04 column-index tie rule.
                if minv[col] < delta {
                    delta = minv[col];
                    col1 = col;
                }
            }
            for col in 0..=m {
                if used[col] {
                    u[matched_row[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = col1;
            if matched_row[col0] == 0 {
                break;
            }
        }
        loop {
            let col1 = way[col0];
            matched_row[col0] = matched_row[col1];
            col0 = col1;
            if col0 == 0 {
                break;
            }
        }
    }

    let mut selected = Vec::new();
    for (col1, &row1) in matched_row
        .iter()
        .enumerate()
        .take(request.n_jobs + 1)
        .skip(1)
    {
        if row1 != 0 && best.contains_key(&(row1 - 1, col1 - 1)) {
            selected.push((row1 - 1, col1 - 1));
        }
    }
    Ok(matching_key(selected))
}

fn matching_score(request: &JointRequest, matching: &Matching) -> Result<f64, String> {
    // `_matching_score` uses a dict comprehension: the last duplicate edge for
    // a pair wins, independently of Hungarian's best-pair selection rule.
    let mut scores = BTreeMap::new();
    for edge in &request.edges {
        scores.insert((edge.unit, edge.job), edge.base_score);
    }
    let mut total = 0.0;
    for pair in matching {
        total += scores
            .get(pair)
            .ok_or_else(|| format!("matching pair {pair:?} has no edge score"))?;
    }
    Ok(total)
}

fn symmetric_changed_count(left: &Matching, right: &Matching) -> usize {
    let left: BTreeSet<Pair> = left.iter().copied().collect();
    let right: BTreeSet<Pair> = right.iter().copied().collect();
    left.symmetric_difference(&right).count() / 2
}

/// Enumerate the RQ61 joint matching rows from one crop-modified R04 graph.
pub fn compile_joint_rows(request: &JointRequest) -> Result<Vec<JointRow>, String> {
    validate(request)?;
    let baseline = matching_key(
        request
            .baseline_matching
            .iter()
            .map(|pair| (pair[0], pair[1])),
    );
    let baseline_pairs: BTreeSet<Pair> = baseline.iter().copied().collect();
    let baseline_score = matching_score(request, &baseline)?;

    let mut single_reachable: BTreeSet<Matching> = BTreeSet::from([baseline.clone()]);
    // unit -> (score, -index semantics represented by the smaller index, index, key)
    let mut best_by_unit: BTreeMap<usize, (f64, usize, Matching)> = BTreeMap::new();
    for edge in &request.edges {
        if baseline_pairs.contains(&(edge.unit, edge.job)) {
            continue;
        }
        let key = decode_matching(request, &[edge.index])?;
        single_reachable.insert(key.clone());
        if key == baseline {
            continue;
        }
        let score = matching_score(request, &key)?;
        let replace = match best_by_unit.get(&edge.unit) {
            None => true,
            Some((old_score, old_index, _)) => {
                score > *old_score || (score == *old_score && edge.index < *old_index)
            }
        };
        if replace {
            best_by_unit.insert(edge.unit, (score, edge.index, key));
        }
    }

    let representatives: Vec<usize> = best_by_unit.values().map(|(_, index, _)| *index).collect();
    let mut decoded: BTreeMap<Matching, JointRow> = BTreeMap::new();
    for left_position in 0..representatives.len() {
        for right_position in (left_position + 1)..representatives.len() {
            let left = representatives[left_position];
            let right = representatives[right_position];
            let first = &request.edges[left];
            let second = &request.edges[right];
            if first.unit == second.unit || first.job == second.job {
                continue;
            }
            let key = decode_matching(request, &[left, right])?;
            if key == baseline || single_reachable.contains(&key) {
                continue;
            }
            let heuristic_score = matching_score(request, &key)?;
            let row = JointRow {
                forced_edge_indices: vec![left, right],
                forced_pairs: vec![[first.unit, first.job], [second.unit, second.job]],
                forced_base_scores: vec![first.base_score, second.base_score],
                matching: key.iter().map(|pair| [pair.0, pair.1]).collect(),
                heuristic_score,
                rank: 0,
                matching_size: key.len(),
                changed_count: symmetric_changed_count(&baseline, &key),
                heuristic_delta: heuristic_score - baseline_score,
            };
            let replace = decoded
                .get(&key)
                .is_none_or(|old| row.forced_edge_indices < old.forced_edge_indices);
            if replace {
                decoded.insert(key, row);
            }
        }
    }

    let mut novel: Vec<JointRow> = decoded.into_values().collect();
    novel.sort_by(|left, right| {
        right
            .heuristic_score
            .partial_cmp(&left.heuristic_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.forced_edge_indices.cmp(&right.forced_edge_indices))
    });
    for (rank, row) in novel.iter_mut().enumerate() {
        row.rank = rank;
    }
    Ok(novel)
}

/// Time only the native graph compilation loop. JSON parsing/serialization and
/// Python/ctypes overhead are intentionally outside the timed region so the
/// caller can report kernel and wrapper paths separately.
pub fn benchmark_joint_rows(request: &JointBenchRequest) -> Result<JointBenchResult, String> {
    if request.requests.is_empty() {
        return Err("joint benchmark needs at least one graph".to_string());
    }
    if request.repetitions == 0 {
        return Err("joint benchmark repetitions must be positive".to_string());
    }
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..request.repetitions {
        for graph in &request.requests {
            let rows = compile_joint_rows(graph)?;
            checksum = checksum.wrapping_add(rows.len());
            for row in rows {
                checksum = checksum
                    .wrapping_add(row.matching_size)
                    .wrapping_add(row.forced_edge_indices.iter().sum::<usize>());
            }
        }
    }
    Ok(JointBenchResult {
        graph_calls: request.requests.len() * request.repetitions,
        elapsed_ns: started.elapsed().as_nanos(),
        checksum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(index: usize, score: f64, unit: usize, job: usize) -> EdgeRecord {
        EdgeRecord {
            index,
            base_score: score,
            unit,
            job,
            extra_walk: 0,
        }
    }

    #[test]
    fn forced_residual_and_pair_basis_produce_a_single_unreachable_matching() {
        let request = JointRequest {
            n_units: 3,
            n_jobs: 3,
            edges: vec![
                edge(0, 10.0, 0, 0),
                edge(1, 5.0, 0, 1),
                edge(2, 5.0, 0, 2),
                edge(3, 5.0, 1, 0),
                edge(4, 10.0, 1, 1),
                edge(5, 5.0, 1, 2),
                edge(6, 5.0, 2, 0),
                edge(7, 5.0, 2, 1),
                edge(8, 10.0, 2, 2),
            ],
            baseline_matching: vec![[0, 0], [1, 1], [2, 2]],
        };
        let rows = compile_joint_rows(&request).unwrap();
        assert!(!rows.is_empty());
        assert!(
            rows.iter()
                .all(|row| row.matching != request.baseline_matching)
        );
    }

    #[test]
    fn missing_real_edges_leave_units_idle() {
        let request = JointRequest {
            n_units: 3,
            n_jobs: 2,
            edges: vec![edge(0, 4.0, 0, 0), edge(1, -2.0, 1, 1)],
            baseline_matching: vec![[0, 0]],
        };
        assert_eq!(decode_matching(&request, &[]).unwrap(), vec![(0, 0)]);
        assert!(compile_joint_rows(&request).unwrap().is_empty());
    }

    #[test]
    fn equal_representative_scores_keep_the_smaller_edge_index() {
        let request = JointRequest {
            n_units: 2,
            n_jobs: 3,
            edges: vec![
                edge(0, 10.0, 0, 0),
                edge(1, 5.0, 0, 1),
                edge(2, 5.0, 0, 2),
                edge(3, 10.0, 1, 1),
                edge(4, 9.0, 1, 0),
                edge(5, 9.0, 1, 2),
            ],
            baseline_matching: vec![[0, 0], [1, 1]],
        };
        let rows = compile_joint_rows(&request).unwrap();
        if let Some(row) = rows.first() {
            assert!(row.forced_edge_indices[0] <= row.forced_edge_indices[1]);
        }
    }
}
