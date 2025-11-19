//! Module containing the `generate_sdf_rtree_bvh` function.

use bvh::bvh::Bvh;
use itertools::Itertools;

use crate::{geo, Point, Topology};

use super::rtree::PointWrapper;

/// `RtreeBvhNode` is a node for the r-tree and bvh acceleration structures.
#[derive(Clone)]
pub struct RtreeBvhNode<V: Point> {
    vertices: (V, V, V),
    vertex_indices: (usize, usize, usize),
    bounding_box: (V, V),
    node_index: usize,
}

impl<V: Point> rstar::RTreeObject for RtreeBvhNode<V> {
    type Envelope = rstar::AABB<PointWrapper<V>>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_corners(
            PointWrapper(self.bounding_box.0),
            PointWrapper(self.bounding_box.1),
        )
    }
}

impl<V: Point> rstar::PointDistance for RtreeBvhNode<V> {
    // Required method
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        geo::point_triangle_distance2(
            &point.0,
            &self.vertices.0,
            &self.vertices.1,
            &self.vertices.2,
        )
    }
}

impl<V: Point> bvh::aabb::Bounded<f32, 3> for RtreeBvhNode<V> {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        let min = nalgebra::Point3::new(
            self.bounding_box.0.x(),
            self.bounding_box.0.y(),
            self.bounding_box.0.z(),
        );
        let max = nalgebra::Point3::new(
            self.bounding_box.1.x(),
            self.bounding_box.1.y(),
            self.bounding_box.1.z(),
        );
        bvh::aabb::Aabb::with_bounds(min, max)
    }
}

impl<V: Point> bvh::bounding_hierarchy::BHShape<f32, 3> for RtreeBvhNode<V> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// Acceleration structure for the r-tree and bvh method.
///
/// Stores the r-tree and bvh acceleration structures.
/// Used to query the sdf for a given query point via [`query_sdf_rtree_bvh`].
#[derive(Clone)]
pub struct SdfAccelerationRtreeBvh<V: Point> {
    pub rtree: rstar::RTree<RtreeBvhNode<V>>,
    pub bvh: Bvh<f32, 3>,
    pub bvh_nodes: Vec<RtreeBvhNode<V>>,
    pub vertices: Vec<V>,
}

/// Generate a signed distance field from a mesh using an r-tree for nearest neighbor search and a bvh for ray intersection.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
pub fn generate_sdf_rtree_bvh<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
) -> Vec<f32>
where
    V: Point + 'static,
    I: Copy + Into<u32> + Sync + Send,
{
    build_sdf_acceleration_rtree_bvh(vertices, indices)
        .map_or_else(std::vec::Vec::new, |acceleration| query_sdf_rtree_bvh(&acceleration, query_points))
}

pub fn build_sdf_acceleration_rtree_bvh<V, I>(
    vertices: &[V],
    indices: Topology<I>,
) -> Option<SdfAccelerationRtreeBvh<V>>
where
    V: Point + 'static,
    I: Copy + Into<u32> + Sync + Send,
{
    let rtree_bvh_nodes = Topology::get_triangles(vertices, indices)
    .map(|triangle| RtreeBvhNode {
        vertices: (
            vertices[triangle.0],
            vertices[triangle.1],
            vertices[triangle.2],
        ),
        vertex_indices: triangle,
        node_index: 0,
        bounding_box: geo::triangle_bounding_box(
            &vertices[triangle.0],
            &vertices[triangle.1],
            &vertices[triangle.2],
        ),
    })
    .collect_vec();

    if rtree_bvh_nodes.is_empty() {
        return None;
    }

    // Build both acceleration structures sequentially
    let mut bvh_nodes = rtree_bvh_nodes.clone();
    let bvh = Bvh::build(&mut bvh_nodes);
    let rtree = rstar::RTree::bulk_load(rtree_bvh_nodes);

    let acceleration: SdfAccelerationRtreeBvh<V> = SdfAccelerationRtreeBvh {
        rtree,
        bvh,
        bvh_nodes,
        vertices: vertices.to_vec(),
    };

    Some(acceleration)
}

pub fn query_sdf_rtree_bvh<V>(
    acceleration: &SdfAccelerationRtreeBvh<V>,
    query_points: &[V],
) -> Vec<f32>
where
    V: Point + 'static,
{
    query_points
        .iter()
        .map(|point| {
            // Query multiple candidate triangles and select the one with minimum distance.
            // This handles cases where closest_point_triangle returns incorrect projections
            // for certain triangle orientations, ensuring robust distance computation.
            const K_NEAREST: usize = 5;
            
            let nearest_triangles: Vec<_> = acceleration.rtree
                .nearest_neighbor_iter(&PointWrapper(*point))
                .take(K_NEAREST)
                .collect();

            let mut min_dist = f32::MAX;
            for nearest in &nearest_triangles {
                let dist = geo::point_triangle_distance(
                    point,
                    &nearest.vertices.0,
                    &nearest.vertices.1,
                    &nearest.vertices.2,
                );
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            let dist = min_dist;

            let alignments = [
                (geo::GridAlign::X, nalgebra::Vector3::new(1.0, 0.0, 0.0)),
                (geo::GridAlign::Y, nalgebra::Vector3::new(0.0, 1.0, 0.0)),
                (geo::GridAlign::Z, nalgebra::Vector3::new(0.0, 0.0, 1.0)),
            ];

            let mut insides = 0;
            for (alignment, direction) in alignments {
                let ray = bvh::ray::Ray::new(
                    nalgebra::Point3::new(point.x(), point.y(), point.z()),
                    direction,
                );
                let mut intersection_count = 0;
                let mut seen = std::collections::HashSet::new();
                let hitcast = acceleration.bvh.traverse(&ray, &acceleration.bvh_nodes);
                for bvh_node in hitcast {
                    let a = &bvh_node.vertices.0;
                    let b = &bvh_node.vertices.1;
                    let c = &bvh_node.vertices.2;
                    let intersect = geo::ray_triangle_intersection_aligned(
                        point,
                        [a, b, c],
                        alignment,
                        Some(bvh_node.vertex_indices),
                        Some(&mut seen),
                    );
                    if intersect.is_some() {
                        intersection_count += 1;
                    }
                }

                if intersection_count % 2 == 1 {
                    insides += 1;
                }
            }

            // Return inside if at least two are insides.
            if insides > 1 {
                -dist
            } else {
                dist
            }
        })
        .collect()
}

/// Query vectors from query points to the closest points on the nearest triangles.
///
/// Similar to `query_sdf_rtree_bvh`, but instead of returning signed distances,
/// this function returns vectors from each query point to the closest point on
/// the nearest triangle surface, along with the signedness.
///
/// # Arguments
/// * `acceleration` - The prebuilt R-tree and BVH acceleration structure
/// * `query_points` - Array of points to query
///
/// # Returns
/// A vector of tuples, where each tuple contains:
/// - A vector pointing from the query point to the closest point on the nearest triangle surface
/// - A boolean indicating signedness (true if inside the mesh, false if outside)
pub fn query_vector_to_closest_point_rtree_bvh<V>(
    acceleration: &SdfAccelerationRtreeBvh<V>,
    query_points: &[V],
) -> Vec<(V, bool)>
where
    V: Point + 'static,
{
    query_points
        .iter()
        .map(|point| {
            // Query multiple candidate triangles to handle incorrect projections from closest_point_triangle.
            // For points near surfaces, average all equidistant results to ensure symmetric query points
            // get symmetric results (critical for physics simulations where asymmetry causes spurious torques).
            const K_NEAREST: usize = 15;
            const RELATIVE_TOLERANCE: f32 = 0.001;
            
            let nearest_triangles: Vec<_> = acceleration.rtree
                .nearest_neighbor_iter(&PointWrapper(*point))
                .take(K_NEAREST)
                .collect();

            // Find minimum distance across all candidate triangles
            let mut min_dist_sq = f32::MAX;
            for nearest in &nearest_triangles {
                let closest_point = geo::closest_point_triangle(
                    point,
                    &nearest.vertices.0,
                    &nearest.vertices.1,
                    &nearest.vertices.2,
                );
                let dist_sq = point.dist2(&closest_point);
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                }
            }
            
            // Average all results within tolerance to eliminate arbitrary triangle selection bias
            let mut sum_point = V::new(0.0, 0.0, 0.0);
            let mut count = 0;
            let tolerance = min_dist_sq * RELATIVE_TOLERANCE + 1e-12;
            
            for nearest in &nearest_triangles {
                let closest_point = geo::closest_point_triangle(
                    point,
                    &nearest.vertices.0,
                    &nearest.vertices.1,
                    &nearest.vertices.2,
                );
                let dist_sq = point.dist2(&closest_point);
                
                if dist_sq <= min_dist_sq + tolerance {
                    sum_point = sum_point.add(&closest_point);
                    count += 1;
                }
            }
            
            let closest_point = if count > 0 {
                let averaged = sum_point.fmul(1.0 / count as f32);
                
                // For points very close to surface, project onto dominant coordinate plane
                // to eliminate floating-point errors in barycentric coordinate arithmetic.
                let dist = min_dist_sq.sqrt();
                if dist < 1e-3 {
                    let vec_to_surface = averaged.sub(point);
                    let abs_x = vec_to_surface.x().abs();
                    let abs_y = vec_to_surface.y().abs();
                    let abs_z = vec_to_surface.z().abs();
                    
                    if abs_z > abs_x && abs_z > abs_y {
                        V::new(point.x(), point.y(), averaged.z())
                    } else if abs_y > abs_x {
                        V::new(point.x(), averaged.y(), point.z())
                    } else {
                        V::new(averaged.x(), point.y(), point.z())
                    }
                } else {
                    averaged
                }
            } else {
                *point
            };

            // Determine signedness using the same ray casting logic as query_sdf_rtree_bvh
            let alignments = [
                (geo::GridAlign::X, nalgebra::Vector3::new(1.0, 0.0, 0.0)),
                (geo::GridAlign::Y, nalgebra::Vector3::new(0.0, 1.0, 0.0)),
                (geo::GridAlign::Z, nalgebra::Vector3::new(0.0, 0.0, 1.0)),
            ];

            let mut insides = 0;
            for (alignment, direction) in alignments {
                let ray = bvh::ray::Ray::new(
                    nalgebra::Point3::new(point.x(), point.y(), point.z()),
                    direction,
                );
                let mut intersection_count = 0;
                let mut seen = std::collections::HashSet::new();
                let hitcast = acceleration.bvh.traverse(&ray, &acceleration.bvh_nodes);
                for bvh_node in hitcast {
                    let a = &bvh_node.vertices.0;
                    let b = &bvh_node.vertices.1;
                    let c = &bvh_node.vertices.2;
                    let intersect = geo::ray_triangle_intersection_aligned(
                        point,
                        [a, b, c],
                        alignment,
                        Some(bvh_node.vertex_indices),
                        Some(&mut seen),
                    );
                    if intersect.is_some() {
                        intersection_count += 1;
                    }
                }

                if intersection_count % 2 == 1 {
                    insides += 1;
                }
            }

            // Determine if inside (at least two directions vote for inside)
            let is_inside = insides > 1;

            // Return the vector from query point to closest point on triangle and signedness
            (closest_point.sub(point), is_inside)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{generate_grid_sdf, generate_sdf, AccelerationMethod, Grid, SignMethod};

    use super::*;

    const EXAMPLE_FULL_MESH_INDICES: &[u32] = &[0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
    const EXAMPLE_FULL_MESH_VERTICES: &[[f32; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];


    #[test]
    fn test_generate_rtree_bvh() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();
        let query_points = [
            cgmath::Vector3::new(0.01, 0.01, 0.5),
            cgmath::Vector3::new(1.0, 1.0, 1.0),
            cgmath::Vector3::new(0.1, 0.2, 0.2),
            cgmath::Vector3::new(1.1, 2.2, 5.2),
            cgmath::Vector3::new(-0.1, 0.2, -0.2),
        ];

        let rtree_sdf = generate_sdf_rtree_bvh(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
        );

        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::Bvh(SignMethod::Raycast),
        );

        let acceleration = build_sdf_acceleration_rtree_bvh(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
        );
        assert!(acceleration.is_some());
        let acceleration = acceleration.unwrap();

        for (idx, (rtree, sdf)) in rtree_sdf.iter().zip(sdf.iter()).enumerate() {
            assert!(
                (rtree - sdf).abs() < 0.01,
                "{:?}: {} != {}",
                query_points[idx],
                rtree,
                sdf
            );
        }
        // Same check but for the acceleration structure.
        let sdf_acceleration = query_sdf_rtree_bvh(&acceleration, &query_points);
        for (sdf, baseline) in sdf_acceleration.iter().zip(rtree_sdf.iter()) {
            assert!(sdf == baseline, "{sdf} != {baseline}"); // should be identical - exact same algorithm
        }
    }

    #[test]
    fn test_query_vector_to_closest_point_sign_distance() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();
        let query_points = [
            cgmath::Vector3::new(0.01, 0.01, 0.5),
            cgmath::Vector3::new(1.0, 1.0, 1.0),
            cgmath::Vector3::new(0.1, 0.2, 0.2),
        ];

        let acceleration = build_sdf_acceleration_rtree_bvh(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
        );
        assert!(acceleration.is_some());
        let acceleration = acceleration.unwrap();

        let results = query_vector_to_closest_point_rtree_bvh(&acceleration, &query_points);

        // Test that we get the expected number of results
        assert_eq!(results.len(), query_points.len());

        // Test that the distance computed from the vector matches the SDF distance
        let sdf_distances = query_sdf_rtree_bvh(&acceleration, &query_points);

        for (i, ((vector, is_inside), sdf_distance)) in results.iter().zip(sdf_distances.iter()).enumerate() {
            let vector_magnitude = vector.length();

            // The magnitude of the vector should match the absolute value of the SDF distance
            assert!(
                (vector_magnitude - sdf_distance.abs()).abs() < 0.001,
                "Point {}: vector magnitude {} doesn't match SDF distance magnitude {}",
                i, vector_magnitude, sdf_distance.abs()
            );

            // The signedness should match the SDF sign
            let sdf_is_inside = *sdf_distance < 0.0;
            assert_eq!(
                *is_inside, sdf_is_inside,
                "Point {i}: signedness {is_inside} doesn't match SDF sign (SDF: {sdf_distance})"
            );
        }
    }

    #[test]
    fn test_query_vector_to_closest_point_rtree_bvh_known_points() {
        const EXAMPLE_FULL_MESH_INDICES: &[u32] = &[0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
        const EXAMPLE_FULL_MESH_VERTICES: &[[f32; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];
        let points_to_test: [([f32; 3], [f32; 3], bool); 7] = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], false), // on a corner
            ([0.0, 0.5, 0.0], [0.0, 0.0, 0.0], false), // on an edge
            ([0.2, 0.2, 0.0], [0.0, 0.0, 0.0], false), // on a face
            ([-0.1, -0.1, 0.2], [0.1, 0.1, -0.2], false), // off a corner
            ([0.1, -0.1, 0.2], [0.0, 0.1, -0.2], false), // off an edge
            ([0.2, 0.2, 0.2], [0.0, 0.0, -0.2], false), // off a face
            ([0.2, 0.2, -0.1], [0.0, 0.0, 0.1], true), // inside (will go to nearest face)
        ];

        let vertices = EXAMPLE_FULL_MESH_VERTICES;
        let indices = EXAMPLE_FULL_MESH_INDICES;

        let acceleration = build_sdf_acceleration_rtree_bvh(
            vertices,
            crate::Topology::TriangleList(Some(indices)),
        );
        assert!(acceleration.is_some());
        let acceleration = acceleration.unwrap();

        for (point, expected_vector, expected_is_inside) in &points_to_test {
            let result = query_vector_to_closest_point_rtree_bvh(&acceleration, &[*point]);
            assert_eq!(result.len(), 1, "Point {point:?} should have one result");
            let vector = result[0].0;
            let is_inside = result[0].1;
            // convert to cgmath::Vector3 for comparison
            let expected_vector = cgmath::Vector3::new(expected_vector[0], expected_vector[1], expected_vector[2]);
            let vector = cgmath::Vector3::new(vector[0], vector[1], vector[2]);
            assert!((vector - expected_vector).length() < 1e-6, "Point {point:?} vector {vector:?} doesn't match expected {expected_vector:?}");
            if vector.length() > 1e-6 { // sign only makes sense when we're not on the surface
                assert_eq!(is_inside, *expected_is_inside, "Point {point:?} is_inside {is_inside:?} doesn't match expected {expected_is_inside:?}");
            }
        }
    }

    #[test]
    fn test_generate_rtree_bvh_big() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();

        let bbox_min = vertices.iter().fold(
            cgmath::Vector3::new(f32::MAX, f32::MAX, f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.min(v.x),
                y: acc.y.min(v.y),
                z: acc.z.min(v.z),
            },
        );
        let bbox_max = vertices.iter().fold(
            cgmath::Vector3::new(-f32::MAX, -f32::MAX, -f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.max(v.x),
                y: acc.y.max(v.y),
                z: acc.z.max(v.z),
            },
        );

        let grid = Grid::from_bounding_box(&bbox_min, &bbox_max, [32, 32, 32]);
        let mut query_points = Vec::new();
        for x in 0..grid.get_cell_count()[0] {
            for y in 0..grid.get_cell_count()[1] {
                for z in 0..grid.get_cell_count()[2] {
                    query_points.push(grid.get_cell_center(&[x, y, z]));
                }
            }
        }
        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::RtreeBvh,
        );
        let grid_sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &grid,
            SignMethod::Raycast,
        );

        // Test against generate_sdf
        for (i, (sdf, grid_sdf)) in sdf.iter().zip(grid_sdf.iter()).enumerate() {
            assert!(
                (sdf - grid_sdf).abs() < 0.01,
                "i: {i}: {sdf} {grid_sdf}"
            );
        }
    }
}
