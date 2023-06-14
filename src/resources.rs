use std::io::{BufReader, Cursor};

use nalgebra as na;

use crate::{model, texture};
use cfg_if::cfg_if;
use eyre::Result;
use wgpu::util::DeviceExt;

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> Result<reqwest::Url> {
    let window = web_sys::window().ok_or_else(|| eyre::eyre!("could not find the window object"))?;
    let location = window.location();
    let base = reqwest::Url::parse(&format!(
        "{}/{}/",
        location
            .origin()
            .map_err(|_| eyre::eyre!("could not get the location origin"))?,
        option_env!("RES_PATH").unwrap_or("res")
    ))?;
    base.join(file_name).map_err(Into::into)
}

pub async fn load_string(file_name: &str) -> Result<String> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name)?;
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
            let txt = std::fs::read_to_string(path)?;
        }
    }

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> Result<Vec<u8>> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name)?;
            let data = reqwest::get(url)
                .await?
                .bytes()
                .await?
                .to_vec();
        } else {
            let path = std::path::Path::new(env!("OUT_DIR"))
                .join("res")
                .join(file_name);
            let data = std::fs::read(path)?;
        }
    }

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    is_normal_map: bool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name, is_normal_map)
}

pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_reader = &mut BufReader::new(Cursor::new(obj_text));

    let (models, obj_materials) = tobj::load_obj_buf_async(
        obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();

    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture, false, device, queue).await?;
        let normal_texture = load_texture(&m.normal_texture, true, device, queue).await?;

        materials.push(model::Material::new(
            device,
            &m.name,
            diffuse_texture,
            normal_texture,
            layout,
        ));
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            #[derive(Copy, Clone)]
            struct Vectors {
                tangent: na::Vector3<f32>,
                bitangent: na::Vector3<f32>,
            }
            let vertices = &mut (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                    tangent: [0.0; 4],
                })
                .collect::<Vec<_>>();

            let indices = &m.mesh.indices;

            // stores the sum of all calculated vectors
            let mut tb_sum = vec![
                Vectors {
                    tangent: na::Vector3::zeros(),
                    bitangent: na::Vector3::zeros(),
                };
                vertices.len()
            ];

            for c in indices.chunks(3) {
                let v0 = vertices[c[0] as usize];
                let v1 = vertices[c[1] as usize];
                let v2 = vertices[c[2] as usize];

                let pos0: na::Vector3<f32> = v0.position.into();
                let pos1: na::Vector3<f32> = v1.position.into();
                let pos2: na::Vector3<f32> = v2.position.into();
                let uv0: na::Vector2<f32> = v0.tex_coords.into();
                let uv1: na::Vector2<f32> = v1.tex_coords.into();
                let uv2: na::Vector2<f32> = v2.tex_coords.into();

                // calculate edges
                let e1 = pos1 - pos0;
                let e2 = pos2 - pos0;

                // calculate texture direction
                let tx1 = uv1 - uv0;
                let tx2 = uv2 - uv0;

                // the tangent and bitangent vectors are determined by the matrix equation:
                //
                //       __           __                               __           __
                //      |               |                             |               |
                //      |               |   __               __       |               |
                //      |    ->  ->     |  |                   |      |    ->  ->     |
                //      |    t   b      |  |     ->     ->     |   =  |    e1  e2     |
                //      |               |  |     tx1    tx2    |      |               |
                //      |               |  |                   |      |               |
                //      |__           __|  |__               __|      |__           __|
                //
                //
                // so we just need to multiply the rhs by the inverse of the middle matrix to obtain
                // the tangent and bitangent vectors.
                let e = na::Matrix3x2::from_columns(&[e1, e2]);
                let tx = na::Matrix2::from_columns(&[tx1, tx2]);

                let tb_mat = e * tx.try_inverse().expect("inverse should be calculable");
                let t = tb_mat.column(0);
                // invert b to enable right-handed normal maps in wgpu
                let b = -tb_mat.column(1);

                tb_sum[c[0] as usize].tangent += t;
                tb_sum[c[1] as usize].tangent += t;
                tb_sum[c[2] as usize].tangent += t;
                tb_sum[c[0] as usize].bitangent += b;
                tb_sum[c[1] as usize].bitangent += b;
                tb_sum[c[2] as usize].bitangent += b;
            }

            for (i, vertex) in vertices.iter_mut().enumerate() {
                let normal: &na::Vector3<f32> = &vertex.normal.into();
                let tan_sum = &tb_sum[i].tangent;
                let bitan_sum = &tb_sum[i].bitangent;

                // Gram-Schmidt orthonormalization
                let tan = &(tan_sum - tan_sum.dot(&normal) * normal).normalize();

                // Store the sign to reconstruct the bitangent in the shader
                let sign = tan_sum.cross(bitan_sum).dot(normal).signum();

                vertex.tangent = na::Vector::push(tan, sign).into();
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}
