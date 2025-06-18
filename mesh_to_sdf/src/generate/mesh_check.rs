//! Mesh checking - used to check that the mesh is watertight and has no duplicate vertices or faces.

use crate::Point;
use std::collections::HashSet;
use core::hash::Hash;

///
/// This is a simple check that the mesh is watertight and has no duplicate vertices or faces.
/// It does not check that the normals are pointing outwards.
///
/// You may want to use this check if you want to use the `[SignMethod::Raycast]` method
/// since it requires the mesh to be watertight.
///
/// # Arguments
///
/// * `vertices` - The vertices of the mesh.
/// * `faces` - The faces of the mesh.
///
/// # Returns
///
/// * `Ok(())` - If the mesh is watertight and has no duplicate vertices or faces.
/// * `Err(String)` - Otherwise.
///
pub fn check_mesh_triangle_list<V, U>(vertices: &[V], faces: &[U]) -> Result<(), String>
where
    V: Copy + Point,
    U: Copy + Eq + Hash + core::fmt::Debug,
{
    // Check for correct length of faces
    if faces.len() % 3 != 0 {
        return Err("Faces must be a multiple of 3".to_string());
    }

    let faces_vec: Vec<&[U]> = faces.chunks(3).collect();

    // Check for duplicate vertices using string representation since f32 doesn't implement Eq/Hash
    let mut seen_vertices = HashSet::new();
    if let Some(duplicate_vertex) = vertices
        .iter()
        .find(|v| !seen_vertices.insert(format!("{:?}_{:?}_{:?}", v.x(), v.y(), v.z())))
    {
        return Err(format!("Duplicate vertex found at ({:?}, {:?}, {:?})", duplicate_vertex.x(), duplicate_vertex.y(), duplicate_vertex.z()));
    }

    // Check for duplicate faces using vector comparison
    let mut seen_faces = HashSet::new();
    if let Some(duplicate_face) = faces_vec.iter().find(|f| !seen_faces.insert(f.to_vec())) {
        return Err(format!("Duplicate face found: {duplicate_face:?}"));
    }

    // Now, make sure that our faces are all in the same direction by making sure that for every edge A -> B, we also have one going B -> A
    let mut edges = Vec::new();
    for face in faces_vec {
        let edge_1 = (face[0], face[1]);
        let edge_2 = (face[1], face[2]);
        let edge_3 = (face[2], face[0]);
        edges.push(edge_1);
        edges.push(edge_2);
        edges.push(edge_3);
    }
    #[allow(clippy::manual_while_let_some)] // we do want a full 'while' loop here, not a 'while let'
    while !edges.is_empty() {
        let edge = edges.pop().unwrap();
        let opposite_edge = (edge.1, edge.0);
        let edge_count = edges.iter().filter(|e| **e == edge).count();
        if edge_count > 0 {
            return Err(format!(
                "Duplicate edge {:?} -> {:?} - do you have an inverted normal?",
                edge.0, edge.1
            ));
        }
        if !edges.contains(&opposite_edge) {
            return Err(format!("Mesh is not waterproof - edge {:?} -> {:?} does not have a corresponding edge {:?} -> {:?}", edge.0, edge.1, edge.1, edge.0));
        }
        // also pop off opposite edge
        edges.retain(|e| *e != opposite_edge);
    }

    // NB: we haven't actually checked that the normals are all pointing outwards. Sounds hard.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_FULL_MESH_INDICES: &[u32] = &[0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
    const EXAMPLE_FULL_MESH_VERTICES: &[[f32; 3]] = &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]];

    #[test]
    fn test_check_mesh_triangle_list_ok() {
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let faces = EXAMPLE_FULL_MESH_INDICES.to_vec();

        let result = check_mesh_triangle_list(&vertices, &faces);
        assert!(
            result.is_ok(),
            "{}",
            format!("Expected mesh to be ok, but got '{}'", result.unwrap_err())
        );
    }

    #[test]
    fn test_check_mesh_triangle_list_duplicate_vertices() {
        let mut vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let last_vertex = vertices[vertices.len()-1];
        vertices.push(last_vertex); // add a duplicate vertex (not even using it, but still want to fail this check)
        let faces = EXAMPLE_FULL_MESH_INDICES.to_vec();

        let result = check_mesh_triangle_list(&vertices, &faces);
        assert!(
            result.is_err(),
            "Expected mesh to be err, but was ok"
        );
    }

    #[test]
    fn test_check_mesh_triangle_list_duplicate_faces() {
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let mut faces = EXAMPLE_FULL_MESH_INDICES.to_vec();
        let last_three: Vec<u32> = faces[faces.len()-3..].to_vec();
        faces.extend_from_slice(&last_three); // make the last face turn up twice

        let result = check_mesh_triangle_list(&vertices, &faces);
        assert!(
            result.is_err(),
            "Expected mesh to be err, but was ok"
        );
    }

    #[test]
    fn test_check_mesh_triangle_list_not_waterproof() {
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();
        let mut faces = EXAMPLE_FULL_MESH_INDICES.to_vec();
        let final_length = faces.len().saturating_sub(3);
        faces.truncate(final_length); // remove the last face

        let result = check_mesh_triangle_list(&vertices, &faces);
        assert!(
            result.is_err(),
            "Expected mesh to be err, but was ok"
        );
    }

    #[test]
    fn test_check_mesh_triangle_list_reversed_normal() {
        let mut indices = EXAMPLE_FULL_MESH_INDICES.to_vec();
        indices[0] = EXAMPLE_FULL_MESH_INDICES[2]; // reverse the normal of the last face
        indices[1] = EXAMPLE_FULL_MESH_INDICES[1];
        indices[2] = EXAMPLE_FULL_MESH_INDICES[0];
        let vertices = EXAMPLE_FULL_MESH_VERTICES.to_vec();

        let result = check_mesh_triangle_list(&vertices, &indices);
        assert!(
            result.is_err(),
            "Expected mesh to be err, but was ok"
        );
    }
}