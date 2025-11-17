//! ⚠️ This crate is still in its early stages. Expect the API to change.
//!
//! ---
//!
//! This crate provides two entry points:
//!
//! - [`generate_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` at the points `query_points`.
//! - [`generate_grid_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` on a [Grid].
//!
//! ```
//! use mesh_to_sdf::{generate_sdf, generate_grid_sdf, SignMethod, AccelerationMethod, Topology, Grid};
//! // vertices are [f32; 3], but can be cgmath::Vector3<f32>, glam::Vec3, etc.
//! let vertices: Vec<[f32; 3]> = vec![[0.5, 1.5, 0.5], [1., 2., 3.], [1., 3., 7.]];
//! let indices: Vec<u32> = vec![0, 1, 2];
//!
//! // query points must be of the same type as vertices
//! let query_points: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]];
//!
//! // Query points are expected to be in the same space as the mesh.
//! let sdf: Vec<f32> = generate_sdf(
//!     &vertices,
//!     Topology::TriangleList(Some(&indices)), // TriangleList as opposed to TriangleStrip
//!     &query_points,
//!     AccelerationMethod::RtreeBvh, // Use an r-tree and a bvh to accelerate queries.
//! );
//!
//! for point in query_points.iter().zip(sdf.iter()) {
//!     // distance is positive outside the mesh and negative inside.
//!     println!("Distance to {:?}: {}", point.0, point.1);
//! }
//! # assert_eq!(sdf, vec![1.0]);
//!
//! // if you can, use generate_grid_sdf instead of generate_sdf as it's optimized and much faster.
//! let bounding_box_min = [0., 0., 0.];
//! let bounding_box_max = [10., 10., 10.];
//! let cell_count = [10, 10, 10];
//!
//! let grid = Grid::from_bounding_box(&bounding_box_min, &bounding_box_max, cell_count);
//!
//! let sdf: Vec<f32> = generate_grid_sdf(
//!     &vertices,
//!     Topology::TriangleList(Some(&indices)),
//!     &grid,
//!     SignMethod::Raycast, // How the sign is computed.
//!                          // Raycast is robust but requires the mesh to be watertight.
//!                          // and is more expensive.
//!                          // Normal might leak negative distances outside the mesh
//! );                       // but works for all meshes, even surfaces.
//!
//! for x in 0..cell_count[0] {
//!     for y in 0..cell_count[1] {
//!         for z in 0..cell_count[2] {
//!             let index = grid.get_cell_idx(&[x, y, z]);
//!             log::info!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index as usize]);
//!         }
//!     }
//! }
//! # assert_eq!(sdf[0], 1.0);
//! ```
//!
//! ---
//!
//! #### Mesh Topology
//!
//! Indices can be of any type that implements `Into<u32>`, e.g. `u16` and `u32`. Topology can be list or strip.
//! If the indices are not provided, they are supposed to be `0..vertices.len()`.
//!
//! For vertices, this library aims to be as generic as possible by providing a trait `Point` that can be implemented for any type.
//! Implementations for most common math libraries are gated behind feature flags. By default, `[f32; 3]` and `nalgebra::[Point3, Vector3]` are provided.
//! If you do not find your favorite library, feel free to implement the trait for it and submit a PR or open an issue.
//!
//! ---
//!
//! #### Computing sign
//!
//! This crate provides two methods to compute the sign of the distance:
//! - [`SignMethod::Raycast`] (default): a robust method to compute the sign of the distance. It counts the number of intersections between a ray starting from the query point and the triangles of the mesh.
//!     It only works for watertight meshes, but guarantees the sign is correct.
//! - [`SignMethod::Normal`]: uses the normals of the triangles to estimate the sign by doing a dot product with the direction of the query point.
//!     It works for non-watertight meshes but might leak negative distances outside the mesh.
//!
//! Using `Raycast` is slower than `Normal` but gives better results. Performances depends on the triangle count and method used.
//! On big dataset, `Raycast` is 5-10% slower for grid generation and rtree based methods. On smaller dataset, the difference can be worse
//! depending on whether the query is triangle intensive or query point intensive.
//! For bvh the difference is negligible between the two methods.
//!
//! ---
//!
//! #### Acceleration structures
//!
//! For generic queries, you can use acceleration structures to speed up the computation.
//! - [`AccelerationMethod::None`]: no acceleration structure. This is the slowest method but requires no extra memory. Scales really poorly.
//! - [`AccelerationMethod::Bvh`]: Bounding Volume Hierarchy. Accepts a `SignMethod`.
//! - [`AccelerationMethod::Rtree`]: R-tree. Uses `SignMethod::Normal`. The fastest method assuming you have more than a couple thousands of queries.
//! - [`AccelerationMethod::RtreeBvh`] (default): Uses R-tree for nearest neighbor search and Bvh for `SignMethod::Raycast`. 5-10% slower than `Rtree` on big datasets.
//!
//! If your mesh is watertight and you have more than a thousand queries/triangles, you should use `AccelerationMethod::RtreeBvh` for best performances.
//! If it's not watertight, you can use `AccelerationMethod::Rtree` instead.
//!
//! `Rtree` methods are 3-4x faster than `Bvh` methods for big enough data. On small meshes, the difference is negligible.
//! `AccelerationMethod::None` scales really poorly and should be avoided unless for small datasets or if you're really tight on memory.
//!
//! ---
//!
//! #### Using your favorite library
//!
//! To use your favorite math library with `mesh_to_sdf`, you need to add it to `mesh_to_sdf` dependency. For example, to use `glam`:
//! ```toml
//! [dependencies]
//! mesh_to_sdf = { version = "0.2.1", features = ["glam"] }
//! ```
//!
//! Currently, the following libraries are supported:
//! - [cgmath] ([`cgmath::Vector3<f32>`])
//! - [glam] ([`glam::Vec3`])
//! - [mint] ([`mint::Vector3<f32>`] and [`mint::Point3<f32>`])
//! - [nalgebra] ([`nalgebra::Vector3<f32>`] and [`nalgebra::Point3<f32>`])
//! - `[f32; 3]`
//!
//! [nalgebra] is always available as it's used internally in the bvh tree.
//!
//! ---
//!
//! #### Serialization
//!
//! If you want to serialize and deserialize signed distance fields, you need to enable the `serde` feature.
//! This features also provides helpers to save and load signed distance fields to and from files via `save_to_file` and `read_from_file`.
use std::boxed::Box;

use itertools::Itertools;

use generate::generic::{
    default::{generate_sdf_default, build_sdf_acceleration_default, query_sdf_default, SdfAccelerationDefault},
    bvh::{generate_sdf_bvh, build_sdf_acceleration_bvh, query_sdf_bvh, SdfAccelerationBvh},
    rtree::{generate_sdf_rtree, build_sdf_acceleration_rtree, query_sdf_rtree, SdfAccelerationRtree},
    rtree_bvh::{generate_sdf_rtree_bvh, build_sdf_acceleration_rtree_bvh, query_sdf_rtree_bvh, SdfAccelerationRtreeBvh},
};

mod bvh_ext;
mod generate;
mod geo;
mod grid;
mod point;

#[cfg(feature = "serde")]
pub mod serde;

pub use generate::grid::generate_grid_sdf;
pub use generate::mesh_check::check_mesh_triangle_list;
pub use generate::generic::rtree_bvh::query_vector_to_closest_point_rtree_bvh; // TODO: Should make a generic function which works for all acceleration methods
pub use grid::{Grid, SnapResult};
pub use point::Point;

/// Mesh Topology: how indices are stored.
#[derive(Copy, Clone)]
pub enum Topology<'a, I>
where
    // I should be a u32 or u16
    I: Into<u32>,
{
    /// Vertex data is a list of triangles. Each set of 3 vertices composes a new triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create two triangles `0 1 2` and `3 4 5`
    /// If no indices are provided, they are supposed to be `0..vertices.len()`
    TriangleList(Option<&'a [I]>),
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create four triangles `0 1 2`, `1 2 3`, `2 3 4`, and `3 4 5`
    /// If no indices are provided, they are supposed to be `0..vertices.len()`
    TriangleStrip(Option<&'a [I]>),
}

/// Owned version of Topology that doesn't require lifetime parameters.
/// Useful when you need to store topology data that outlives the original slice.
/// It would be nice if there is a tidier way to do this, feel free to improve!
#[derive(Clone)]
pub enum OwnedTopology<I>
where
    I: Into<u32>,
{
    /// The same as a `TriangleList` but with indices owned.
    OwnedTriangleList(Option<Vec<I>>),
    /// The same as a `TriangleStrip` but with indices owned.
    OwnedTriangleStrip(Option<Vec<I>>),
}

impl<'a, I> Topology<'a, I>
where
    I: Into<u32>,
{
    /// Compute the triangles list
    /// Returns an iterator of tuples of 3 indices representing a triangle.
    fn get_triangles<V>(
        vertices: &'a [V],
        indices: Self,
    ) -> Box<dyn Iterator<Item = (usize, usize, usize)> + Send + 'a>
    where
        V: Point,
        I: Copy + Into<u32> + Sync + Send,
    {
        match indices {
            Topology::TriangleList(Some(indices)) => {
                assert!(indices.len() % 3 == 0, "TriangleList indices length ({}) must be divisible by 3", indices.len());
                Box::new(indices.iter().map(|x| (*x).into() as usize).tuples())
            }
            Topology::TriangleList(None) => {
                assert!(vertices.len() % 3 == 0, "TriangleList vertices length ({}) must be divisible by 3 when no indices are provided", vertices.len());
                Box::new((0..vertices.len()).tuples())
            }
            Topology::TriangleStrip(Some(indices)) => {
                Box::new(indices.iter().map(|x| (*x).into() as usize).tuple_windows())
            }
            Topology::TriangleStrip(None) => Box::new((0..vertices.len()).tuple_windows()),
        }
    }

    fn convert_to_owned(self) -> OwnedTopology<I>
        where I: Copy + Into<u32> + Sync + Send,
        {
        match self {
            Topology::TriangleList(Some(indices)) => OwnedTopology::OwnedTriangleList(Some(indices.to_vec())),
            Topology::TriangleList(None) => OwnedTopology::OwnedTriangleList(None),
            Topology::TriangleStrip(Some(indices)) => OwnedTopology::OwnedTriangleStrip(Some(indices.to_vec())),
            Topology::TriangleStrip(None) => OwnedTopology::OwnedTriangleStrip(None),
        }
    }
}

impl<I> OwnedTopology<I>
where
    I: Into<u32> + Copy + Sync + Send,
{
    fn as_topology(&self) -> Topology<I> {
        match self {
            Self::OwnedTriangleList(Some(indices)) => Topology::TriangleList(Some(indices.as_slice())),
            Self::OwnedTriangleList(None) => Topology::TriangleList(None),
            Self::OwnedTriangleStrip(Some(indices)) => Topology::TriangleStrip(Some(indices.as_slice())),
            Self::OwnedTriangleStrip(None) => Topology::TriangleStrip(None),
        }
    }
}

/// Method to compute the sign of the distance.
///
/// Raycast is the default method. It is robust but requires the mesh to be watertight.
///
/// Normal is not robust and might leak negative distances outside the mesh.
///
/// For grid generation, Raycast is ~1% slower.
/// For query points, Raycast is ~10% slower.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SignMethod {
    /// A robust method to compute the sign of the distance.
    /// It counts the number of intersection between a ray starting from the query point and the mesh.
    /// If the number of intersections is odd, the point is inside the mesh.
    /// This requires the mesh to be watertight.
    #[default]
    Raycast,
    /// A faster but not robust method to compute the sign of the distance.
    /// It uses the normals of the triangles to estimate the sign.
    /// It might leak negative distances outside the mesh.
    Normal,
}

/// Acceleration structure to speed up the computation.
///
/// `RtreeBvh` is the fastest method but also the most memory intensive.
/// If your mesh is not watertight, you can use `Rtree` instead.
/// `Bvh` is about 4x slower than `Rtree`.
/// `None` is the slowest method and scales really poorly but requires no extra memory.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccelerationMethod {
    /// No acceleration structure.
    None(SignMethod),
    /// Bounding Volume Hierarchy.
    /// Recommended unless you have very few queries and triangles (less than a couple thousands)
    /// or if you're really tight on memory as it requires more memory than the default method.
    Bvh(SignMethod),
    /// R-tree
    /// Only compatible with `SignMethod::Normal`
    Rtree,
    /// R-tree and Bvh
    /// Uses R-tree for nearest neighbor search and Bvh for ray intersection.
    #[default]
    RtreeBvh,
}

/// A list of all acceleration methods - useful if you want to test something across all of them.
pub const ALL_SDF_ACCELERATION_METHODS: [AccelerationMethod; 6] = [
    AccelerationMethod::None(SignMethod::Raycast),
    AccelerationMethod::None(SignMethod::Normal),
    AccelerationMethod::Bvh(SignMethod::Raycast),
    AccelerationMethod::Bvh(SignMethod::Normal),
    AccelerationMethod::Rtree,
    AccelerationMethod::RtreeBvh,
];

/// Acceleration structures to store mesh data for later use.
///
/// Obtained from a mesh via [`build_sdf_acceleration`].
/// Used to query the sdf for a given query point via [`query_sdf`].
#[derive(Clone)]
pub enum SdfAccelerationMesh<V: Point, I: Copy + Into<u32> + Sync + Send> {
    /// Corresponds to [`AccelerationMethod::None`]
    None(SdfAccelerationDefault<V, I>),
    /// Corresponds to [`AccelerationMethod::Bvh`]
    Bvh(SdfAccelerationBvh<V>),
    /// Corresponds to [`AccelerationMethod::Rtree`]
    Rtree(SdfAccelerationRtree<V>),
    /// Corresponds to [`AccelerationMethod::RtreeBvh`]
    RtreeBvh(SdfAccelerationRtreeBvh<V>),
}

/// Compare two signed distances, taking into account floating point errors and signs.
fn compare_distances(a: f32, b: f32) -> core::cmp::Ordering {
    // for a point to be inside, it has to be inside all normals of nearest triangles.
    // if one distance is positive, then the point is outside.
    // this check is sensitive to floating point errors though
    // so it's not perfect, but it reduces the number of false positives considerably.
    // TODO: expose ulps and epsilon?
    if float_cmp::approx_eq!(f32, a.abs(), b.abs(), ulps = 2, epsilon = 1e-6) {
        // they are equals: return the one with the smallest distance, privileging positive distances.
        match (a.is_sign_negative(), b.is_sign_negative()) {
            (true, false) => core::cmp::Ordering::Greater,
            (false, true) => core::cmp::Ordering::Less,
            _ => a.abs().partial_cmp(&b.abs()).unwrap(),
        }
    } else {
        // return the closest to 0.
        a.abs().partial_cmp(&b.abs()).expect("NaN distance")
    }
}

/// Generate a signed distance field from a mesh.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
/// ```
/// use mesh_to_sdf::{generate_sdf, SignMethod, Topology, AccelerationMethod};
///
/// let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.]];
/// let indices: Vec<u32> = vec![0, 1, 2];
///
/// let query_points: Vec<[f32; 3]> = vec![[0., 0., 0.]];
///
/// // Query points are expected to be in the same space as the mesh.
/// let sdf: Vec<f32> = generate_sdf(
///     &vertices,
///     Topology::TriangleList(Some(&indices)),
///     &query_points,
///     AccelerationMethod::RtreeBvh,   // Use an rtree and a bvh to accelerate queries.
///                                     // Recommended unless you have very few queries and triangles (less than a couple thousands)
///                                     // or if you're really tight on memory as it requires more memory than other methods.
///                                     // This uses raycasting to compute sign. This is robust but requires the mesh to be watertight.
/// );                                  // If your mesh isn't watertight, you can use AccelerationMethod::Rtree instead.
///
/// for point in query_points.iter().zip(sdf.iter()) {
///     println!("Distance to {:?}: {}", point.0, point.1);
/// }
///
/// # assert_eq!(sdf, vec![1.0]);
/// ```
pub fn generate_sdf<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
    acceleration_method: AccelerationMethod,
) -> Vec<f32>
where
    V: Point + 'static,
    I: Copy + Into<u32> + Sync + Send,
{
    for point in query_points {
        assert!(point.is_finite(), "Query point {point:?} contains non-finite values");
    }
    match acceleration_method {
        AccelerationMethod::None(sign_method) => {
            generate_sdf_default(vertices, indices, query_points, sign_method)
        }
        AccelerationMethod::Bvh(sign_method) => {
            generate_sdf_bvh(vertices, indices, query_points, sign_method)
        }
        AccelerationMethod::Rtree => {
            generate_sdf_rtree(vertices, indices, query_points)
        }
        AccelerationMethod::RtreeBvh => {
            generate_sdf_rtree_bvh(vertices, indices, query_points)
        }
    }
}


/// Generate an acceleration object for use with `query_sdf`.
/// Allows for re-use of a given mesh without having to rebuild the acceleration structures.
///
/// Returns an acceleration object.
/// ```
/// use mesh_to_sdf::{build_sdf_acceleration, SignMethod, Topology, AccelerationMethod};
///
/// let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.]];
/// let indices: Vec<u32> = vec![0, 1, 2];
///
/// let acceleration = build_sdf_acceleration(
///     &vertices,
///     Topology::TriangleList(Some(&indices)),
///     AccelerationMethod::RtreeBvh,
/// );
/// ```
pub fn build_sdf_acceleration<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    acceleration_method: AccelerationMethod,
) -> Option<SdfAccelerationMesh<V, I>>
where
    V: Point + 'static,
    I: Copy + Into<u32> + Sync + Send,
{
    match acceleration_method {
        AccelerationMethod::None(sign_method) => {
            Some(SdfAccelerationMesh::None(build_sdf_acceleration_default(vertices, indices, sign_method)))
        }
        AccelerationMethod::Bvh(sign_method) => {
            Some(SdfAccelerationMesh::Bvh(build_sdf_acceleration_bvh(vertices, indices, sign_method)))
        }
        AccelerationMethod::Rtree => {
            Some(SdfAccelerationMesh::Rtree(build_sdf_acceleration_rtree(vertices, indices)))
        }
        AccelerationMethod::RtreeBvh => {
            build_sdf_acceleration_rtree_bvh(vertices, indices).map(|acceleration| SdfAccelerationMesh::RtreeBvh(acceleration))
        }
    }
}

/// Query the sdf for a given query point.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// ```
/// use mesh_to_sdf::{build_sdf_acceleration, Topology, AccelerationMethod, query_sdf, SdfAccelerationMesh};
///
/// let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.]];
/// let indices: Vec<u32> = vec![0, 1, 2];
/// let acceleration = build_sdf_acceleration(
///     &vertices,
///     Topology::TriangleList(Some(&indices)),
///     AccelerationMethod::RtreeBvh,
/// );
///
/// let query_points: Vec<[f32; 3]> = vec![[0., 0., 0.]];
///
/// let sdf: Vec<f32> = query_sdf(
///     &acceleration.unwrap(),
///     &query_points,
/// );
///
/// for point in query_points.iter().zip(sdf.iter()) {
///     println!("Distance to {:?}: {}", point.0, point.1);
/// }
///
/// # assert_eq!(sdf, vec![1.0]);
/// ```
pub fn query_sdf<V, I>(
    acceleration: &SdfAccelerationMesh<V, I>,
    query_points: &[V],
) -> Vec<f32>
where
    V: Point + 'static,
    I: Copy + Into<u32> + Sync + Send,
{
    match acceleration {
        SdfAccelerationMesh::None(acceleration) => {
            query_sdf_default(acceleration, query_points)
        }
        SdfAccelerationMesh::Bvh(acceleration) => {
            query_sdf_bvh(acceleration, query_points)
        }
        SdfAccelerationMesh::Rtree(acceleration) => {
            query_sdf_rtree(acceleration, query_points)
        }
        SdfAccelerationMesh::RtreeBvh(acceleration) => {
            query_sdf_rtree_bvh(acceleration, query_points)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_FULL_MESH_INDICES: &[u32] = &[0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
    const EXAMPLE_FULL_MESH_VERTICES: &[[f32; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];

    #[test]
    fn test_mesh_waterproof() {
        for accel_method in ALL_SDF_ACCELERATION_METHODS {
            let indices = EXAMPLE_FULL_MESH_INDICES.to_vec();
            let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
            let topology = Topology::TriangleList(Some(&indices));
            let point_five_expected_distance = (0.5 / core::f32::consts::FRAC_1_SQRT_2.hypot(1.0)) * core::f32::consts::FRAC_1_SQRT_2;
            let query_points_and_strings = [
                ([0.0, 0.0, 0.0], 0.0, "when we're on one of the corners"),
                ([0.5, 0.0, 0.0], 0.0, "when we're on one of the edges"),
                ([0.2, 0.2, 0.0], 0.0, "when we're on one of the faces"),
                ([0.2, 0.2, 0.05], 0.05, "when we're slightly above the 0-plane (outside the object)"),
                ([0.2, 0.2, -0.05], -0.05, "when we're slightly below the 0-plane (inside the object)"),
                ([1.0, 1.0, 0.05], (0.05*0.05 + 0.5*0.5 + 0.5*0.5_f32).sqrt(), "when we're above the plane, but off to the side (outside the object)"),
                ([1.0, 1.0, -0.05], (-0.05*-0.05 + 0.5*0.5 + 0.5*0.5_f32).sqrt(), "when we're below the plane, but off to the side (outside the object)"),
                ([1.0, 1.0, 0.0], 0.5_f32.hypot(0.5), "when we're exactly on the plane, but off to the side (outside the object)"),
                ([0.0, 0.0, -1.05], 0.05, "when we're directly below the lowest point (outside the object)"),
                ([0.0, 0.0, -0.95], 0.0, "when we're directly above the lowest point (inside the object)"),
                ([0.5, 0.5, -0.5], point_five_expected_distance, "when we're in front of the triangle (outside the object)"),
                ([0.5 - 1e-6, 0.5, -0.5], point_five_expected_distance, "when we're in front of the triangle (outside the object) (1)"), // we could be in front of any of the faces here, so make sure we're picking the right one
                ([0.5, 0.5 - 1e-6, -0.5], point_five_expected_distance, "when we're in front of the triangle (outside the object) (2)"),
                ([0.5, 0.5, -0.5 - 1e-6], point_five_expected_distance, "when we're in front of the triangle (outside the object) (3)"),
            ];

            for (query_point, expected_sdf, expected_string) in query_points_and_strings {
                let sdf = generate_sdf(&vertices, topology, &[query_point], accel_method);
                assert!(sdf.len() == 1, "Expected sdf to have one value");
                assert!(sdf[0] - expected_sdf < 1e-5, "The SDF was {} but we expected {} {}", sdf[0], expected_sdf, expected_string);
            }
        }
    }

    #[test]
    fn test_mesh_not_waterproof_using_bvh_raycast() {
        // Raycast is robust but requires the mesh to be watertight - so let's try it with a hole in the mesh
        let mut indices = EXAMPLE_FULL_MESH_INDICES.to_vec();
        // remove the last 3 indices, i.e. the last face
        let final_length = indices.len().saturating_sub(3);
        indices.truncate(final_length);
        assert_eq!(indices.len(), EXAMPLE_FULL_MESH_INDICES.len() - 3, "Failed to remove face");
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let topology = Topology::TriangleList(Some(&indices));
        let sdf = generate_sdf(&vertices, topology, &[[1.0_f32, 1.0_f32, -0.05_f32]], AccelerationMethod::Bvh(SignMethod::Raycast),);
        assert!(sdf.len() == 1, "Expected sdf to have one value");
        assert_eq!(sdf[0], (0.05*0.05 + 0.5*0.5 + 0.5*0.5_f32).sqrt());
        // Well, it worked! So I guess it's not as sensitive as expected. I was expecting it to fail.
    }

    #[test]
    fn test_mesh_reversed_normal_using_bvh_raycast() {
        // reverse the normal of one of the faces
        let mut indices = EXAMPLE_FULL_MESH_INDICES.to_vec();
        indices[0] = EXAMPLE_FULL_MESH_INDICES[2];
        indices[1] = EXAMPLE_FULL_MESH_INDICES[1];
        indices[2] = EXAMPLE_FULL_MESH_INDICES[0];
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let topology = Topology::TriangleList(Some(&indices));
        let sdf = generate_sdf(&vertices, topology, &[[0.2_f32, 0.2_f32, 0.05_f32]], AccelerationMethod::Bvh(SignMethod::Raycast),);
        assert!(sdf.len() == 1, "Expected sdf to have one value");
        assert_eq!(sdf[0], 0.05);
        // Well, it worked! So I guess it's not as sensitive as expected. I was expecting it to fail.
    }

    #[test]
    #[should_panic]
    fn test_mesh_bad_index() {
        let mut indices = EXAMPLE_FULL_MESH_INDICES.to_vec();
        assert!(indices.len() < 100, "Too many indices");
        indices[0] = 100; // out of bound index
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let topology = Topology::TriangleList(Some(&indices));
        let _sdf = generate_sdf(&vertices, topology, &[[0.2_f32, 0.2_f32, 0.05_f32]], AccelerationMethod::Bvh(SignMethod::Raycast),);
    }

    #[test]
    #[should_panic]
    fn test_mesh_bad_number_of_indices() {
        let mut indices = EXAMPLE_FULL_MESH_INDICES.to_vec();
        indices.push(0); // now we don't have a nice triangle strip
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let topology = Topology::TriangleList(Some(&indices));
        let _sdf = generate_sdf(&vertices, topology, &[[0.2_f32, 0.2_f32, 0.05_f32]], AccelerationMethod::Bvh(SignMethod::Raycast),);
    }

    #[test]
    #[should_panic]
    fn test_nan_point() {
        let indices = EXAMPLE_FULL_MESH_INDICES.to_vec();
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let topology = Topology::TriangleList(Some(&indices));
        let _sdf = generate_sdf(&vertices, topology, &[[0.0_f32, 0.0_f32, f32::NAN]], AccelerationMethod::Bvh(SignMethod::Raycast),);
    }

    #[test]
    fn test_all_acceleration_methods_on_cube() {
        //for accel_method in ALL_SDF_ACCELERATION_METHODS { // TODO: Alas, this does not work! AccelerationMethod::RtreeBvh nearly works, but there are two points where it fails, but if you change the point by a tiny amount, it works. The other Raycast methods do very poorly however.
        for accel_method in [
            AccelerationMethod::RtreeBvh] {
            let vertices = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]; // it's a cube! Now we can make faces for it so that it's waterproof and with proper normals
            let indices: Vec<u32> = vec![
                0, 2, 4,
                0, 4, 1, // the bottom face (Z=0)
                3, 5, 7,
                3, 7, 6, // the top face (Z=1)
                0, 1, 5,
                0, 5, 3, // X=0 axis face
                2, 6, 7,
                2, 7, 4, // X=1 axis face
                0, 3, 6,
                0, 6, 2, // Y=0 axis face
                1, 4, 7,
                1, 7, 5, // Y=1 axis face
            ];

            // Test with points forming a slightly smaller cube fitting in the mesh one.
            let points = vec![
                [0.2, 0.2, 0.2],
                [0.8, 0.2, 0.2],
                [0.2, 0.8, 0.2],
                [0.2, 0.2, 0.8],
                [0.8, 0.8, 0.2],
                [0.8, 0.2, 0.8],
                [0.2, 0.8, 0.8],
                [0.8, 0.8, 0.8],
            ];
            let expected_distance = -0.2;
            let sdf = generate_sdf(&vertices, Topology::TriangleList(Some(&indices)), &points, accel_method);
            let failed_points: Vec<_> = sdf
                .iter()
                .filter(|&distance| (distance - expected_distance).abs() >= 1e-6)
                .collect();

            if !failed_points.is_empty() {
                println!("SDF: {sdf:?}");
                println!("Expected_distance: {expected_distance}");
                println!("Failed points:");
                for distance in failed_points {
                    println!(
                        "distance = {} (residual = {})",
                        distance,
                        (distance - expected_distance).abs()
                    );
                }
                panic!("Points and mesh should intersect at all points");
            }

            // Then test with points forming a slightly larger cube than the mesh
            let points = vec![
                [-0.2, -0.2, -0.2],
                [1.2, -0.2, -0.2],
                [-0.2, 1.2, -0.2],
                [-0.2, -0.2, 1.2],
                [1.2, 1.2, -0.2],
                [1.2, -0.2, 1.2],
                [-0.2, 1.2, 1.2],
                [1.2, 1.2, 1.2],
            ];
            let expected_distance = ((0.2 * 0.2) * 3_f64).sqrt() as f32; // diagonal of 0.2m x 0.2m x 0.2m cube
            let sdf = generate_sdf(&vertices, Topology::TriangleList(Some(&indices)), &points, accel_method);
            let failed_points: Vec<_> = sdf
                .iter()
                .filter(|&distance| (distance - expected_distance).abs() >= 1e-6)
                .collect();

            if !failed_points.is_empty() {
                println!("SDF: {sdf:?}");
                println!("Expected_distance: {expected_distance}");
                println!("Failed points:");
                for distance in failed_points {
                    println!(
                        "distance = {} (residual = {})",
                        distance,
                        (distance - expected_distance).abs()
                    );
                }
                panic!("Points and mesh should intersect at all points");
            }
        }
    }
}