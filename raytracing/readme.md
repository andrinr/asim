# Shading

- surface normal ``n``
- viewer direction ``v``
- light direction ``l``
- reflection direction ``r`` (in angle = out angle)

inlcude metallic, then specular color is equal to the light source color
otherwise the spec light is equal to the material color

add refraction as well
theres total internal reflection as well. Happen if in angle (relative to normal) is greater than n_2 / n_1. Where n_1 is the refractive index of the material and n_2 is the refractive index of the medium the ray is entering.

Fresnel equations, otherwise we can use schlick approximation


## Triangulations

Ray triangle intersection: Given 3 vertices of a triangle $v_0, v_1, v_2$, normals $n_1, n_2, n_3$ and a ray $r = o + td$, find the intersection point $p$ if it exists. Take dot product between some normal and 


$\hat{n} \cdot v_i  + D - 0$ plane equation
$D = - \hat{n} \cdot v_i$

$\hat{n} \cdot p - ( \hat{n} \cdot v_i + D) = 0$ then p is on the plane.

$n = (v_1 - v_0) \times (v_2 - v_0)$

Considering ray $r = o + td$, we can write the equation of the plane as:

$$ (\hat{n} \cdot d) t +  (\hat{n} + o) + D = 0$$

solve for t:

$$ t = \frac{-(\hat{n} \cdot o) - D}{(\hat{n} \cdot d)} $$

Check for division by zero (before !)

We then need to check weather the ray intersects with the triangle. 

- Project the triangle from 3D to 2D. 
    - Take largest element of triangle normal and project onto the plane perpendicular to that axis. i.e. ignore the other dimensions of the traingle description.

---

## Baycentric coordinates for the Triangle




