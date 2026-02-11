//! Directed acyclic graph for step dependency resolution.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::WorkflowError;

/// A directed acyclic graph that resolves step dependencies.
///
/// Each node has a string ID and a list of dependency IDs (edges).
/// Supports topological ordering and parallel group computation.
#[derive(Debug, Clone, Default)]
pub struct Dag {
    /// Maps node ID -> list of dependency IDs (incoming edges).
    dependencies: HashMap<String, Vec<String>>,
    /// Insertion order for deterministic output.
    order: Vec<String>,
}

impl Dag {
    /// Creates an empty DAG.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the DAG has no nodes.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Returns the number of nodes.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns `true` if the node exists.
    pub fn contains(&self, id: &str) -> bool {
        self.dependencies.contains_key(id)
    }

    /// Returns the dependencies of a node, or `None` if node doesn't exist.
    pub fn dependencies(&self, id: &str) -> Option<&Vec<String>> {
        self.dependencies.get(id)
    }

    /// Adds a node with its dependencies.
    ///
    /// Returns an error if the node already exists or any dependency is unknown.
    pub fn add_node(&mut self, id: &str, deps: &[&str]) -> Result<(), WorkflowError> {
        if self.dependencies.contains_key(id) {
            return Err(WorkflowError::DuplicateStep {
                step_id: id.into(),
            });
        }

        for dep in deps {
            if !self.dependencies.contains_key(*dep) {
                return Err(WorkflowError::MissingDependency {
                    step_id: id.into(),
                    dependency_id: (*dep).into(),
                });
            }
        }

        let dep_list: Vec<String> = deps.iter().map(|d| (*d).into()).collect();
        self.dependencies.insert(id.into(), dep_list);
        self.order.push(id.into());
        Ok(())
    }

    /// Adds a raw edge (used internally for testing cycle detection).
    #[cfg(test)]
    pub(crate) fn add_edge(&mut self, from: &str, to: &str) {
        if let Some(deps) = self.dependencies.get_mut(from) {
            deps.push(to.into());
        }
    }

    /// Returns a topological ordering of all nodes using Kahn's algorithm.
    ///
    /// Returns an error if a cycle is detected.
    pub fn topological_order(&self) -> Result<Vec<&str>, WorkflowError> {
        if self.is_empty() {
            return Ok(vec![]);
        }

        // Build in-degree map and adjacency (reverse deps: who depends on whom)
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for id in &self.order {
            in_degree.entry(id.as_str()).or_insert(0);
        }

        for (id, deps) in &self.dependencies {
            *in_degree.entry(id.as_str()).or_insert(0) = deps.len();
            for dep in deps {
                dependents
                    .entry(dep.as_str())
                    .or_default()
                    .push(id.as_str());
            }
        }

        // Start with nodes that have no dependencies (in-degree 0)
        let mut queue: VecDeque<&str> = VecDeque::new();
        // Use insertion order for deterministic results
        for id in &self.order {
            if in_degree[id.as_str()] == 0 {
                queue.push_back(id.as_str());
            }
        }

        let mut result = Vec::with_capacity(self.order.len());

        while let Some(node) = queue.pop_front() {
            result.push(node);

            if let Some(deps) = dependents.get(node) {
                // Sort dependents by insertion order for determinism
                let mut sorted_deps: Vec<&str> = deps.clone();
                sorted_deps.sort_by_key(|d| {
                    self.order
                        .iter()
                        .position(|o| o.as_str() == *d)
                        .unwrap_or(usize::MAX)
                });

                for dependent in sorted_deps {
                    let deg = in_degree.get_mut(dependent).expect("node must exist");
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(dependent);
                    }
                }
            }
        }

        if result.len() != self.order.len() {
            // Find cycle participants
            let in_result: HashSet<&str> = result.iter().copied().collect();
            let cycle_nodes: Vec<String> = self
                .order
                .iter()
                .filter(|id| !in_result.contains(id.as_str()))
                .cloned()
                .collect();
            return Err(WorkflowError::CycleDetected { steps: cycle_nodes });
        }

        Ok(result)
    }

    /// Returns groups of nodes that can be executed in parallel.
    ///
    /// Each group contains nodes whose dependencies are all in previous groups.
    pub fn parallel_groups(&self) -> Result<Vec<Vec<&str>>, WorkflowError> {
        if self.is_empty() {
            return Ok(vec![]);
        }

        let topo = self.topological_order()?;
        let mut node_level: HashMap<&str, usize> = HashMap::new();

        for node in &topo {
            let deps = &self.dependencies[*node];
            let level = if deps.is_empty() {
                0
            } else {
                deps.iter()
                    .map(|d| node_level[d.as_str()] + 1)
                    .max()
                    .unwrap_or(0)
            };
            node_level.insert(node, level);
        }

        let max_level = node_level.values().copied().max().unwrap_or(0);
        let mut groups: Vec<Vec<&str>> = vec![vec![]; max_level + 1];
        for node in &topo {
            groups[node_level[node]].push(node);
        }

        Ok(groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkflowError;

    // === Construction ===

    #[test]
    fn empty_dag_has_no_nodes() {
        let dag = Dag::new();
        assert!(dag.is_empty());
        assert_eq!(dag.len(), 0);
    }

    // === Adding Nodes ===

    #[test]
    fn add_node_without_dependencies() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        assert_eq!(dag.len(), 1);
        assert!(!dag.is_empty());
    }

    #[test]
    fn add_node_with_dependencies() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        dag.add_node("b", &["a"]).unwrap();
        assert_eq!(dag.len(), 2);
    }

    #[test]
    fn add_duplicate_node_returns_error() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        let result = dag.add_node("a", &[]);
        assert!(matches!(
            result,
            Err(WorkflowError::DuplicateStep { step_id }) if step_id == "a"
        ));
    }

    #[test]
    fn add_node_with_unknown_dependency_returns_error() {
        let mut dag = Dag::new();
        let result = dag.add_node("b", &["unknown"]);
        assert!(matches!(
            result,
            Err(WorkflowError::MissingDependency { step_id, dependency_id })
            if step_id == "b" && dependency_id == "unknown"
        ));
    }

    // === Topological Sort ===

    #[test]
    fn topological_order_single_node() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        let order = dag.topological_order().unwrap();
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn topological_order_linear_chain() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        dag.add_node("b", &["a"]).unwrap();
        dag.add_node("c", &["b"]).unwrap();
        let order = dag.topological_order().unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn topological_order_diamond_shape() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        dag.add_node("b", &["a"]).unwrap();
        dag.add_node("c", &["a"]).unwrap();
        dag.add_node("d", &["b", "c"]).unwrap();
        let order = dag.topological_order().unwrap();

        // a must come first, d must come last
        assert_eq!(order[0], "a");
        assert_eq!(order[3], "d");
        // b and c can be in either order
        let middle: Vec<&str> = order[1..3].to_vec();
        assert!(middle.contains(&"b"));
        assert!(middle.contains(&"c"));
    }

    #[test]
    fn topological_order_empty_dag() {
        let dag = Dag::new();
        let order = dag.topological_order().unwrap();
        assert!(order.is_empty());
    }

    // === Cycle Detection ===

    #[test]
    fn detects_self_cycle() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        // Manually create a self-cycle by inserting edge
        dag.add_edge("a", "a");
        let result = dag.topological_order();
        assert!(matches!(result, Err(WorkflowError::CycleDetected { .. })));
    }

    // === Parallel Groups ===

    #[test]
    fn parallel_groups_single_node() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        let groups = dag.parallel_groups().unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec!["a"]);
    }

    #[test]
    fn parallel_groups_independent_nodes() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        dag.add_node("b", &[]).unwrap();
        dag.add_node("c", &[]).unwrap();
        let groups = dag.parallel_groups().unwrap();
        // All three can run in parallel
        assert_eq!(groups.len(), 1);
        let mut group = groups[0].clone();
        group.sort();
        assert_eq!(group, vec!["a", "b", "c"]);
    }

    #[test]
    fn parallel_groups_linear_chain() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        dag.add_node("b", &["a"]).unwrap();
        dag.add_node("c", &["b"]).unwrap();
        let groups = dag.parallel_groups().unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec!["a"]);
        assert_eq!(groups[1], vec!["b"]);
        assert_eq!(groups[2], vec!["c"]);
    }

    #[test]
    fn parallel_groups_diamond() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        dag.add_node("b", &["a"]).unwrap();
        dag.add_node("c", &["a"]).unwrap();
        dag.add_node("d", &["b", "c"]).unwrap();
        let groups = dag.parallel_groups().unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec!["a"]);
        let mut mid = groups[1].clone();
        mid.sort();
        assert_eq!(mid, vec!["b", "c"]);
        assert_eq!(groups[2], vec!["d"]);
    }

    // === Dependencies Query ===

    #[test]
    fn dependencies_returns_node_deps() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        dag.add_node("b", &["a"]).unwrap();
        let deps = dag.dependencies("b");
        assert_eq!(deps, Some(&vec!["a".to_string()]));
    }

    #[test]
    fn dependencies_returns_none_for_unknown() {
        let dag = Dag::new();
        assert!(dag.dependencies("unknown").is_none());
    }

    #[test]
    fn contains_checks_node_existence() {
        let mut dag = Dag::new();
        dag.add_node("a", &[]).unwrap();
        assert!(dag.contains("a"));
        assert!(!dag.contains("b"));
    }

    // === Property-Based Tests ===

    mod prop {
        use super::*;
        use proptest::prelude::*;

        /// Strategy: generate a list of (node_index, Vec<dep_indices>) where
        /// dep_indices only reference earlier nodes (forward-only → always acyclic).
        fn arb_acyclic_dag() -> impl Strategy<Value = Vec<(usize, Vec<usize>)>> {
            (1usize..=20).prop_flat_map(|n| {
                let edges: Vec<BoxedStrategy<(usize, Vec<usize>)>> = (0..n)
                    .map(|i| {
                        if i == 0 {
                            Just((i, vec![])).boxed()
                        } else {
                            proptest::collection::vec(0..i, 0..=i.min(4))
                                .prop_map(move |deps| {
                                    let unique: Vec<usize> =
                                        deps.into_iter().collect::<std::collections::HashSet<_>>()
                                            .into_iter().collect();
                                    (i, unique)
                                })
                                .boxed()
                        }
                    })
                    .collect();
                edges
            })
        }

        /// Strategy: generate arbitrary edges that may contain cycles.
        fn arb_maybe_cyclic_edges() -> impl Strategy<Value = Vec<(usize, Vec<usize>)>> {
            (2usize..=15).prop_flat_map(|n| {
                let edges: Vec<BoxedStrategy<(usize, Vec<usize>)>> = (0..n)
                    .map(|i| {
                        proptest::collection::vec(0..n, 0..=3)
                            .prop_map(move |deps| {
                                let unique: Vec<usize> =
                                    deps.into_iter()
                                        .filter(|d| *d != i)
                                        .collect::<std::collections::HashSet<_>>()
                                        .into_iter().collect();
                                (i, unique)
                            })
                            .boxed()
                    })
                    .collect();
                edges
            })
        }

        /// Build a Dag from index-based edges (bypasses add_node validation).
        fn build_dag_from_edges(edges: &[(usize, Vec<usize>)]) -> Dag {
            let mut dag = Dag::new();
            // First pass: add all nodes without deps
            for (i, _) in edges {
                let id = format!("n{i}");
                let _ = dag.add_node(&id, &[]);
            }
            // Second pass: inject dependency edges directly
            for (i, deps) in edges {
                let id = format!("n{i}");
                if let Some(dep_list) = dag.dependencies.get_mut(&id) {
                    for d in deps {
                        let dep_id = format!("n{d}");
                        if !dep_list.contains(&dep_id) {
                            dep_list.push(dep_id);
                        }
                    }
                }
            }
            dag
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]

            #[test]
            fn acyclic_dag_always_produces_valid_topo_order(edges in arb_acyclic_dag()) {
                let dag = build_dag_from_edges(&edges);
                let order = dag.topological_order().unwrap();

                // All nodes present
                prop_assert_eq!(order.len(), dag.len());

                // Every dependency appears before its dependent
                let pos: std::collections::HashMap<&str, usize> = order
                    .iter()
                    .enumerate()
                    .map(|(i, id)| (*id, i))
                    .collect();

                for (id, deps) in &dag.dependencies {
                    for dep in deps {
                        prop_assert!(
                            pos[dep.as_str()] < pos[id.as_str()],
                            "dep {} should come before {} in topo order",
                            dep, id
                        );
                    }
                }
            }

            #[test]
            fn topo_order_never_hangs(edges in arb_maybe_cyclic_edges()) {
                let dag = build_dag_from_edges(&edges);
                let result = dag.topological_order();
                // Must terminate: either Ok or CycleDetected
                let valid = result.is_ok()
                    || matches!(result, Err(WorkflowError::CycleDetected { .. }));
                prop_assert!(valid, "topological_order must return Ok or CycleDetected");
            }

            #[test]
            fn parallel_groups_cover_all_nodes(edges in arb_acyclic_dag()) {
                let dag = build_dag_from_edges(&edges);
                let groups = dag.parallel_groups().unwrap();
                let total: usize = groups.iter().map(Vec::len).sum();
                prop_assert_eq!(total, dag.len());
            }

            #[test]
            fn parallel_groups_respect_dependencies(edges in arb_acyclic_dag()) {
                let dag = build_dag_from_edges(&edges);
                let groups = dag.parallel_groups().unwrap();

                // Build group index for each node
                let mut group_of: std::collections::HashMap<&str, usize> =
                    std::collections::HashMap::new();
                for (gi, group) in groups.iter().enumerate() {
                    for id in group {
                        group_of.insert(id, gi);
                    }
                }

                // Every dependency must be in a strictly earlier group
                for (id, deps) in &dag.dependencies {
                    for dep in deps {
                        prop_assert!(
                            group_of[dep.as_str()] < group_of[id.as_str()],
                            "dep {} (group {}) should be in earlier group than {} (group {})",
                            dep, group_of[dep.as_str()], id, group_of[id.as_str()]
                        );
                    }
                }
            }
        }
    }
}
