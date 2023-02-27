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
