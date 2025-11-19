use crate::point::Point;

/// Compute the bounding box of a triangle.
pub fn triangle_bounding_box<V: Point>(a: &V, b: &V, c: &V) -> (V, V) {
    const EPSILONF: f32 = 0.0001;

    let min = V::new(
        f32::min(a.x(), f32::min(b.x(), c.x())),
        f32::min(a.y(), f32::min(b.y(), c.y())),
        f32::min(a.z(), f32::min(b.z(), c.z())),
    );
    let max = V::new(
        f32::max(a.x(), f32::max(b.x(), c.x())),
        f32::max(a.y(), f32::max(b.y(), c.y())),
        f32::max(a.z(), f32::max(b.z(), c.z())),
    );

    // We add a small epsilon to the max and subtract a small epsilon from the min to avoid
    // floating point errors in the bvh aabb intersection tests and make sure the aabb has a volume.
    let epsilon = V::new(EPSILONF, EPSILONF, EPSILONF);
    (min.sub(&epsilon), max.add(&epsilon))
}

/// Compute the distance between a point and a triangle.
/// The result is always positive.
pub fn point_triangle_distance<V: Point>(x0: &V, x1: &V, x2: &V, x3: &V) -> f32 {
    // Compute the unsigned distance from the point to the plane of the triangle
    let nearest = closest_point_triangle(x0, x1, x2, x3);
    x0.dist(&nearest)
}

/// Compute the distance squared between a point and a triangle.
pub fn point_triangle_distance2<V: Point>(x0: &V, x1: &V, x2: &V, x3: &V) -> f32 {
    // Compute the unsigned distance from the point to the plane of the triangle
    let nearest = closest_point_triangle(x0, x1, x2, x3);
    x0.dist2(&nearest)
}

/// Compute the signed distance between a point and a triangle.
/// Sign is positive if the point is outside the mesh, negative if inside.
/// Assume all normals are pointing outside the mesh.
/// Sign is computed by checking if the point is on the same side of the triangle as the normal.
pub fn point_triangle_signed_distance<V: Point>(x0: &V, x1: &V, x2: &V, x3: &V) -> f32 {
    // Compute the unsigned distance from the point to the plane of the triangle
    let nearest = closest_point_triangle(x0, x1, x2, x3);
    let direction = x0.sub(&nearest);
    let normal = triangle_normal(x1, x2, x3);

    let distance = x0.dist(&nearest);

    if direction.dot(&normal) > 0.0 {
        distance
    } else {
        -distance
    }
}

/// Return the normal, which is NOT normalized.
/// TODO: this might be better with the mesh normals if we add them to the api.
fn triangle_normal<V: Point>(a: &V, b: &V, c: &V) -> V {
    let ab = b.sub(a);
    let ac = c.sub(a);
    ab.cross(&ac)
}

/// Project a point onto a triangle.
/// Adapted from Embree.
/// <https://github.com/embree/embree/blob/master/tutorials/common/math/closest_point.h#L10>
#[expect(clippy::many_single_char_names)]
pub fn closest_point_triangle<V: Point>(p: &V, a: &V, b: &V, c: &V) -> V {
    // Add safety checks for degenerate triangles
    #[allow(clippy::match_same_arms)]
    match (a.eq(b), b.eq(c), a.eq(c)) {
        (true, true, true) => {
            return *a;
        }
        (true, _, _) => {
            return closest_point_segment(p, a, c);
        }
        (_, true, _) => {
            return closest_point_segment(p, a, b);
        }
        (_, _, true) => {
            return closest_point_segment(p, a, b);
        }
        // they are all different
        _ => {}
    }

    // Actual embree code.
    let ab = b.sub(a);
    let ac = c.sub(a);
    let ap = p.sub(a);

    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return *a;
    }

    let bp = p.sub(b);
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return *b;
    }

    let cp = p.sub(c);
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return *c;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a.add(&ab.fmul(v));
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let v = d2 / (d2 - d6);
        return a.add(&ac.fmul(v));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = c.sub(b);
        return b.add(&bc.fmul(v));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a.add(&ab.fmul(v)).add(&ac.fmul(w))
}

/// Project p on \[ab\].
fn closest_point_segment<V: Point>(p: &V, a: &V, b: &V) -> V {
    let ab = b.sub(a);
    let m = ab.dot(&ab);

    // find parameter value of closest point on segment
    let ap = p.sub(a);
    let mut s12 = ab.dot(&ap) / m;
    s12 = s12.clamp(0.0, 1.0);

    a.add(&ab.fmul(s12))
}

/// Grid alignment for raycast.
/// Used exclusively in `ray_triangle_intersection_aligned`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GridAlign {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HitKey {
    Vertex(usize),
    Edge(usize, usize),
}

pub fn ray_triangle_intersection_aligned<V: Point>(
    ray_origin: &V,
    triangle: [&V; 3],
    alignment: GridAlign,
    vertex_indices: Option<(usize, usize, usize)>,
    seen: Option<&mut std::collections::HashSet<HitKey>>,
) -> Option<f32> {
    let edge01 = triangle[1].sub(triangle[0]); // v0->v1
    let edge12 = triangle[2].sub(triangle[1]); // v1->v2
    let edge20 = triangle[0].sub(triangle[2]); // v2->v0

    let p0 = ray_origin.sub(triangle[0]); // v0->ray
    let p1 = ray_origin.sub(triangle[1]); // v1->ray
    let p2 = ray_origin.sub(triangle[2]); // v2->ray

    // While named get_x, get_y, get_z, they are rotated around the grid alignment.
    // x is the ray direction axis. (y, z) is the triangle projection plane.
    let get_y = match alignment {
        GridAlign::X => |v: &V| v.y(),
        GridAlign::Y => |v: &V| v.z(),
        GridAlign::Z => |v: &V| v.x(),
    };
    let get_z = match alignment {
        GridAlign::X => |v: &V| v.z(),
        GridAlign::Y => |v: &V| v.x(),
        GridAlign::Z => |v: &V| v.y(),
    };
    let get_x = match alignment {
        GridAlign::X => |v: &V| v.x(),
        GridAlign::Y => |v: &V| v.y(),
        GridAlign::Z => |v: &V| v.z(),
    };

    // 2D edge-function weights on the (y,z) plane (your original formulation)
    // w0 corresponds to edge v1->v2 (opposite v0)
    // w1 corresponds to edge v2->v0 (opposite v1)
    // w2 corresponds to edge v0->v1 (opposite v2)
    let w0 = get_z(&p1) * get_y(&edge12) - get_y(&p1) * get_z(&edge12);
    let w1 = get_z(&p2) * get_y(&edge20) - get_y(&p2) * get_z(&edge20);
    let w2 = get_z(&p0) * get_y(&edge01) - get_y(&p0) * get_z(&edge01);

    if (w0 < 0.0 && w1 < 0.0 && w2 < 0.0) || (w0 > 0.0 && w1 > 0.0 && w2 > 0.0) {
        let t = -(w0 * get_x(&p0) + w2 * get_x(&p2) + w1 * get_x(&p1)) / (w0 + w1 + w2);
        if t > 0.0 {
            return Some(t);
        }
        return None;
    }

    // ---------- Boundary handling with exact comparisons (no tolerance) ----------

    // Exact “on edge” flags
    let on0 = w0 == 0.0; // on edge v1->v2
    let on1 = w1 == 0.0; // on edge v2->v0
    let on2 = w2 == 0.0; // on edge v0->v1

    // For edge cases, require the other two weights to be non-negative together or non-positive together
    let same_sign_non_strict = |a: f32, b: f32| (a >= 0.0 && b >= 0.0) || (a <= 0.0 && b <= 0.0);

    // Top-left inclusive predicate on projected (y,z) edge direction.
    #[inline]
    fn top_left_inclusive_du_dv(du: f32, dv: f32) -> bool {
        (dv > 0.0) || (dv == 0.0 && du < 0.0)
    }

    // Precompute numerator and denominator for t (same as interior path)
    let t_num = w0 * get_x(&p0) + w2 * get_x(&p2) + w1 * get_x(&p1);
    let t_den = w0 + w1 + w2;

    #[inline]
    fn return_t(t_num: f32, t_den: f32) -> Option<f32> {
        let t = -t_num / t_den;
        if t > 0.0 { Some(t) } else { None }
    }
    
    #[inline]
    fn try_dedup_and_return(
        t_num: f32,
        t_den: f32,
        key: HitKey,
        vertex_indices: Option<(usize, usize, usize)>,
        seen: Option<&mut std::collections::HashSet<HitKey>>,
    ) -> Option<f32> {
        let t = -t_num / t_den;
        if t > 0.0 {
            if let (Some(_), Some(seen_set)) = (vertex_indices, seen) {
                if !seen_set.insert(key) {
                    return None;
                }
            }
            Some(t)
        } else {
            None
        }
    }

    let on_count = (on0 as u8) + (on1 as u8) + (on2 as u8);
    if on_count >= 2 {
        let (vertex_idx, inclusive) = if on2 && on0 {
            let inc01 = top_left_inclusive_du_dv(get_y(&edge01), get_z(&edge01));
            let inc12 = top_left_inclusive_du_dv(get_y(&edge12), get_z(&edge12));
            (1, inc01 || inc12)
        } else if on0 && on1 {
            let inc12 = top_left_inclusive_du_dv(get_y(&edge12), get_z(&edge12));
            let inc20 = top_left_inclusive_du_dv(get_y(&edge20), get_z(&edge20));
            (2, inc12 || inc20)
        } else {
            let inc20 = top_left_inclusive_du_dv(get_y(&edge20), get_z(&edge20));
            let inc01 = top_left_inclusive_du_dv(get_y(&edge01), get_z(&edge01));
            (0, inc20 || inc01)
        };

        if inclusive {
            if let Some(indices) = vertex_indices {
                let v = match vertex_idx {
                    0 => indices.0,
                    1 => indices.1,
                    2 => indices.2,
                    _ => unreachable!(),
                };
                return try_dedup_and_return(t_num, t_den, HitKey::Vertex(v), vertex_indices, seen);
            } else {
                return return_t(t_num, t_den);
            }
        }
        return None;
    }

    if on2 && same_sign_non_strict(w0, w1) {
        if top_left_inclusive_du_dv(get_y(&edge01), get_z(&edge01)) {
            if let Some(indices) = vertex_indices {
                let (v0, v1) = (indices.0, indices.1);
                return try_dedup_and_return(t_num, t_den, HitKey::Edge(v0.min(v1), v0.max(v1)), vertex_indices, seen);
            } else {
                return return_t(t_num, t_den);
            }
        }
        return None;
    }
    if on0 && same_sign_non_strict(w1, w2) {
        if top_left_inclusive_du_dv(get_y(&edge12), get_z(&edge12)) {
            if let Some(indices) = vertex_indices {
                let (v0, v1) = (indices.1, indices.2);
                return try_dedup_and_return(t_num, t_den, HitKey::Edge(v0.min(v1), v0.max(v1)), vertex_indices, seen);
            } else {
                return return_t(t_num, t_den);
            }
        }
        return None;
    }
    if on1 && same_sign_non_strict(w2, w0) {
        if top_left_inclusive_du_dv(get_y(&edge20), get_z(&edge20)) {
            if let Some(indices) = vertex_indices {
                let (v0, v1) = (indices.2, indices.0);
                return try_dedup_and_return(t_num, t_den, HitKey::Edge(v0.min(v1), v0.max(v1)), vertex_indices, seen);
            } else {
                return return_t(t_num, t_den);
            }
        }
        return None;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;
    use proptest::test_runner::Config;

    proptest! {
        #![proptest_config(Config::with_cases(1000))]

        #[test]
        fn test_closest_point_triangle(
            p in prop::array::uniform3(-10.0f32..10.0),
            a in prop::array::uniform3(-10.0f32..10.0),
            b in prop::array::uniform3(-10.0f32..10.0),
            c in prop::array::uniform3(-10.0f32..10.0))
        {
            use float_cmp::approx_eq;

            let cmp = |a: [f32; 3], b: [f32;3]| {
                approx_eq!(f32, a[0], b[0], ulps = 5, epsilon = 1e-3) ||
                approx_eq!(f32, a[1], b[1], ulps = 5, epsilon = 1e-3) ||
                approx_eq!(f32, a[2], b[2], ulps = 5, epsilon = 1e-3)
            };

            if cmp(a, b) || cmp(a, c) || cmp(b, c) {
                // baseline_point_triangle_distance does not work if the triangle is degenerate
                return Ok(());
            }

            let closest = closest_point_triangle(&p, &a, &b, &c);
            let dist = p.dist(&closest);
            let baseline_dist = baseline_point_triangle_distance(&p, &a, &b, &c);

            println!("p: {p:?}, a: {a:?}, b: {b:?}, c: {c:?} - {dist} {baseline_dist}");
            assert!(!dist.is_nan());
            assert!(float_cmp::approx_eq!(f32, dist, baseline_dist, ulps = 5, epsilon = 1e-3));
        }
    }

    proptest! {
        #![proptest_config(Config::with_cases(1000))]

        #[test]
        fn test_ray_triangle_intersection(
            p in prop::array::uniform3(-10.0f32..10.0),
            a in prop::array::uniform3(-10.0f32..10.0),
            b in prop::array::uniform3(-10.0f32..10.0),
            c in prop::array::uniform3(-10.0f32..10.0),
        ) {
            for (align, dir) in [
                (GridAlign::X, [1.0, 0.0, 0.0]),
                (GridAlign::Y, [0.0, 1.0, 0.0]),
                (GridAlign::Z, [0.0, 0.0, 1.0]),
            ] {
                let generic_hit = ray_triangle_intersection_generic(&p, &dir, &a, &b, &c);
                let hit = ray_triangle_intersection_aligned(&p, [&a, &b, &c], align, None, None);

                match (generic_hit, hit) {
                    (None, None) => {}
                    (Some(generic_hit), Some(hit)) => {
                        assert!(float_cmp::approx_eq!(f32, generic_hit, hit, ulps = 5, epsilon = 1e-3), "generic_hit: {generic_hit}, hit: {hit}");
                    }
                    _ => {
                        panic!("generic_hit: {generic_hit:?}, hit: {hit:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_ray_triangle_intersection_generic() {
        let a = [0.0, 1.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 0.0, 1.0];

        let ray_origin = [0.2, 0.2, 0.2];
        let mut ray_dir;

        ray_dir = [0.0, 0.0, 1.0];
        assert!(ray_triangle_intersection_generic(&ray_origin, &ray_dir, &a, &b, &c).is_some());

        ray_dir = [0.0, 0.0, -1.0];
        assert!(ray_triangle_intersection_generic(&ray_origin, &ray_dir, &a, &b, &c).is_none());

        ray_dir = [0.3, 1.0, 0.2];
        assert!(ray_triangle_intersection_generic(&ray_origin, &ray_dir, &a, &b, &c).is_some());

        ray_dir = [0.3, -1.0, -0.2];
        assert!(ray_triangle_intersection_generic(&ray_origin, &ray_dir, &a, &b, &c).is_none());
    }

    #[test]
    fn test_deduplication() {
        use std::collections::HashSet;
        
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        // Test interior hit - not deduplicated
        let interior = [0.3, 0.3, -1.0];
        let mut seen = HashSet::new();
        
        let hit1 = ray_triangle_intersection_aligned(&interior, [&v0, &v1, &v2], GridAlign::Z, Some((0, 1, 2)), Some(&mut seen));
        assert!(hit1.is_some());
        
        let hit2 = ray_triangle_intersection_aligned(&interior, [&v0, &v1, &v2], GridAlign::Z, Some((0, 1, 2)), Some(&mut seen));
        assert!(hit2.is_some(), "Interior hits are not deduplicated");
        assert_eq!(seen.len(), 0, "Interior hits don't add to seen set");
        
        // Test edge hit - should be deduplicated
        // Edge v0->v1 with top-left rule: dv > 0 (vertical edge going up)
        let v0_edge = [0.0, 0.0, 0.0];
        let v1_edge = [0.0, 1.0, 0.0];
        let v2_edge = [1.0, 0.0, 0.0];
        
        let on_edge = [0.0, 0.5, -1.0];  // On edge v0->v1
        let mut seen_edge = HashSet::new();
        
        let hit3 = ray_triangle_intersection_aligned(&on_edge, [&v0_edge, &v1_edge, &v2_edge], GridAlign::Z, Some((0, 1, 2)), Some(&mut seen_edge));
        assert!(hit3.is_some(), "First edge hit should register");
        assert_eq!(seen_edge.len(), 1, "Edge hit should add to seen set");
        assert!(seen_edge.contains(&HitKey::Edge(0, 1)), "Should contain edge (0,1)");
        
        let hit4 = ray_triangle_intersection_aligned(&on_edge, [&v0_edge, &v1_edge, &v2_edge], GridAlign::Z, Some((0, 1, 2)), Some(&mut seen_edge));
        assert!(hit4.is_none(), "Second edge hit should be deduplicated");
        assert_eq!(seen_edge.len(), 1, "Seen set should still have 1 entry");
        assert!(seen_edge.contains(&HitKey::Edge(0, 1)), "Should still contain edge (0,1)");
        
        // Test vertex hit - should be deduplicated
        // Vertex at v1 with both incident edges passing top-left rule
        let v0_vtx = [0.0, 0.0, 0.0];
        let v1_vtx = [1.0, 0.0, 0.0];
        let v2_vtx = [0.5, 1.0, 0.0];
        
        let on_vertex = [1.0, 0.0, -1.0];  // On vertex v1
        let mut seen_vtx = HashSet::new();
        
        let hit5 = ray_triangle_intersection_aligned(&on_vertex, [&v0_vtx, &v1_vtx, &v2_vtx], GridAlign::Z, Some((0, 1, 2)), Some(&mut seen_vtx));
        assert!(hit5.is_some(), "First vertex hit should register");
        assert_eq!(seen_vtx.len(), 1, "Vertex hit should add to seen set");
        assert!(seen_vtx.contains(&HitKey::Vertex(1)), "Should contain vertex 1");
        
        let hit6 = ray_triangle_intersection_aligned(&on_vertex, [&v0_vtx, &v1_vtx, &v2_vtx], GridAlign::Z, Some((0, 1, 2)), Some(&mut seen_vtx));
        assert!(hit6.is_none(), "Second vertex hit should be deduplicated");
        assert_eq!(seen_vtx.len(), 1, "Seen set should still have 1 entry");
        assert!(seen_vtx.contains(&HitKey::Vertex(1)), "Should still contain vertex 1");
    }

    #[test]
    fn test_closest_point_segment() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];

        let p = [0.3, 1.0, 0.0];
        let closest = closest_point_segment(&p, &a, &b);
        assert_eq!(closest, [0.3, 0.0, 0.0]);

        let p = [10.3, 1.0, 10.0];
        let closest = closest_point_segment(&p, &a, &b);
        assert_eq!(closest, [1.0, 0.0, 0.0]);
    }

    /// Find the distance x0 is from triangle x1-x2-x3.
    /// Note: this is adapted from <https://github.com/christopherbatty/SDFGen>
    /// It is slower than the Embree version, but we can use it as a baseline to test the Embree version.
    /// This version does not work for degenerate triangles.
    fn baseline_point_triangle_distance<V: Point>(x0: &V, x1: &V, x2: &V, x3: &V) -> f32 {
        // first find barycentric coordinates of closest point on infinite plane
        let x03 = x0.sub(x3);
        let x13 = x1.sub(x3);
        let x23 = x2.sub(x3);

        let m13 = x13.dot(&x13);
        let m23 = x23.dot(&x23);

        let d = x13.dot(&x23);

        #[allow(clippy::suspicious_operation_groupings)]
        let invdet = 1.0 / f32::max(m13 * m23 - d * d, 1e-30);

        let a = x13.dot(&x03);
        let b = x23.dot(&x03);

        // the barycentric coordinates themselves
        let w23 = invdet * (m23 * a - d * b);
        let w31 = invdet * (m13 * b - d * a);
        let w12 = 1.0 - w23 - w31;

        if w23 >= 0.0 && w31 >= 0.0 && w12 >= 0.0 {
            // if we're inside the triangle
            let x1p = x1.fmul(w23);
            let x2p = x2.fmul(w31);
            let x3p = x3.fmul(w12);
            let projected = x1p.add(&x2p).add(&x3p);
            x0.dist(&projected)
        } else {
            // we have to clamp to one of the edges
            if w23 > 0.0 {
                // this rules out edge 2-3 for us
                f32::min(
                    point_segment_distance(x0, x1, x2),
                    point_segment_distance(x0, x1, x3),
                )
            } else if w31 > 0.0 {
                // this rules out edge 1-3
                f32::min(
                    point_segment_distance(x0, x1, x2),
                    point_segment_distance(x0, x2, x3),
                )
            } else {
                // w12 must be >0, ruling out edge 1-2
                f32::min(
                    point_segment_distance(x0, x1, x3),
                    point_segment_distance(x0, x2, x3),
                )
            }
        }
    }

    /// Find the distance x0 is from segment x1-x2.
    fn point_segment_distance<V: Point>(x0: &V, x1: &V, x2: &V) -> f32 {
        let dx = x2.sub(x1);
        let m2 = dx.dot(&dx);

        // find parameter value of closest point on segment
        let mut s12 = dx.dot(&x2.sub(x0)) / m2;
        s12 = s12.clamp(0.0, 1.0);

        // and find the distance
        x0.dist(&x1.fmul(s12).add(&x2.fmul(1.0 - s12)))
    }

    /// A generic ray-triangle intersection test.
    /// This is a generic version of `ray_triangle_intersection` that works for any ray direction.
    fn ray_triangle_intersection_generic<V: Point>(
        ray_origin: &V,
        ray_dir: &V,
        v0: &V,
        v1: &V,
        v2: &V,
    ) -> Option<f32> {
        // compute the plane's normal
        let v0v1 = v1.sub(v0);
        let v0v2 = v2.sub(v0);
        let n = v0v1.cross(&v0v2);

        // Step 1: finding projected point P

        // check if the ray and plane are parallel.
        let n_dot_ray_dir = n.dot(ray_dir);
        if n_dot_ray_dir.abs() < 0.00001 {
            // almost 0
            return None;
        }

        let d = -n.dot(v0);
        let t = -(n.dot(ray_origin) + d) / n_dot_ray_dir;

        // check if the triangle is behind the ray
        if t < 0.0 {
            return None; // the triangle is behind
        }

        let p = ray_origin.add(&ray_dir.fmul(t));

        // Step 2: inside-outside test

        // edge 0
        let edge0 = v1.sub(v0);
        let vp0 = p.sub(v0);
        let c = edge0.cross(&vp0);
        if n.dot(&c) < 0.0 {
            return None; // P is on the wrong side
        }

        // edge 1
        let edge1 = v2.sub(v1);
        let vp1 = p.sub(v1);
        let c = edge1.cross(&vp1);
        if n.dot(&c) < 0.0 {
            return None; // P is on the wrong side
        }

        // edge 2
        let edge2 = v0.sub(v2);
        let vp2 = p.sub(v2);
        let c = edge2.cross(&vp2);
        if n.dot(&c) < 0.0 {
            return None; // P is on the wrong side;
        }

        Some(t) // this ray hits the triangle
    }
}
