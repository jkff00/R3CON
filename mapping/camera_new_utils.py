
import math
import torch

from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

class SimpleCamera:
    """Bundle camera intrinsics and provide a minimal rendering entry point."""

    def __init__(
        self,
        width = 1600,
        height = 1080,
        fx = 1.0,
        fy = 1.0,
        scale_val = 1.0,
        tag = "perspective",
        device = "cuda"
    ) -> None:
        self.device = torch.device(device)
        self.spherical = tag == "spherical"
        self.bg = torch.tensor([0,0,0.5], dtype=torch.float32, device=device)
        self.width = width
        self.height = height

        if self.spherical:
            self.fovx = math.pi / 2
            self.fovy = math.pi / 2
            self.tanfovx = math.tan(self.fovx * 0.5)
            self.tanfovy = math.tan(self.fovy * 0.5)
            self.fx = self.width/ (2.0 * self.tanfovx)
            self.fy = self.height/ (2.0 * self.tanfovy)
        else:

            if fx<1.0:
                self.fx = width*fx
                self.fy = height*fy
            else:
                self.fx = fx
                self.fy = fy
            self.fovx =2*math.atan(width/(2*self.fx))
            self.fovy = 2*math.atan(height/(2*self.fy))

            self.tanfovx = math.tan(self.fovx * 0.5)
            self.tanfovy = math.tan(self.fovy * 0.5)
                #limit the 2d scale of each Gaussian primative

        self.projectionMatrix = self.getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.fovx, fovY=self.fovy)
        self.visible_thresh = (self.fx*scale_val)**2


    def getProjectionMatrix(self,znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P.transpose(0, 1).to(self.device)
    # Rendering helpers
    def full_proj_transform(self, view_matrix: torch.Tensor) -> torch.Tensor:
        """Compute the combined  w2c view-projection matrix for the active tag."""

        matrix = (view_matrix.unsqueeze(0).bmm(self.projectionMatrix.unsqueeze(0))).squeeze(0)
        return matrix

    def render_simple(self,xyz, opacity, scale, rot, c2w,
                    override_color=None,render_only = False):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        camera_center = c2w[:3, 3]
        w2c = torch.linalg.inv(c2w).T
        full_proj_transform = self.full_proj_transform(w2c)
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, device="cuda") + 0
        raster_settings = GaussianRasterizationSettings(
            image_height=self.height,
            image_width=self.width,
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            bg=self.bg,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj_transform,
            sh_degree=0,
            campos=camera_center,
            prefiltered=False,
            spherical=self.spherical,
            debug=False,
            visible_thresh = self.visible_thresh
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        colors_precomp = override_color 
        rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra,contrib,pixel_contrib = rasterizer(means3D=xyz, means2D=screenspace_points, shs=None,
                                                            colors_precomp=colors_precomp,opacities=opacity,
                                                            scales=scale, rotations=rot,
                                                            cov3Ds_precomp=None)

        pixel_contrib = pixel_contrib.long()
        x = pixel_contrib[:, 0]
        y = pixel_contrib[:, 1]

        mask = (contrib > 0)
    
        if not render_only:
            mask = mask & (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)


        unique_contrib, _,_,index_contrib = self.unique(contrib[mask], dim=0)


        unique_pixel_contrib = pixel_contrib[mask][index_contrib]

        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "contrib":unique_contrib,
                "pixel_contrib":unique_pixel_contrib,
                "depth": rendered_depth}


    def unique(self,x, dim=0):
        unique, inverse, counts = torch.unique(x, dim=dim,
            sorted=True, return_inverse=True, return_counts=True)
        decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
        inv_sorted = (inverse+decimals).argsort()
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]
        # index = index.sort().values
        return unique, inverse, counts, index
    


