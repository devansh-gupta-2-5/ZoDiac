import numpy as np
from scipy.stats import norm
import pywt
import torch
from diffusers.utils.torch_utils import randn_tensor

class GTWatermark():
    def __init__(self, device, shape=(1,4,64,64), dtype=torch.float32, w_channel=3, w_radius=10, generator=None):
        self.device = device
        # from latent tensor
        self.shape = shape
        self.dtype = dtype
        # from hyperparameters
        self.w_channel = w_channel
        self.w_radius = w_radius

        self.gt_patch, self.watermarking_mask = self._gen_gt(generator=generator)
        self.mu, self.sigma = self.watermark_stat()

    def _circle_mask(self, size=64, r=10, x_offset=0, y_offset=0):
        # Reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        return ((x - x0)**2 + (y-y0)**2) <= r**2

    def _get_watermarking_pattern(self, gt_init):
        # Extract the spatial dimensions for wavelet transform
        spatial_dims = (-2, -1)  # Assuming last two dimensions are spatial (height, width)
        
        # Perform wavelet decomposition along spatial dimensions
        coeffs = pywt.wavedec2(gt_init.squeeze().cpu().numpy(), wavelet='haar', level=2, axes=spatial_dims)
        coeffs_array, coeffs_slices = pywt.coeffs_to_array(coeffs)
        
        # Generate a circular mask
        mask = self._circle_mask(coeffs_array.shape[0], self.w_radius)
        
        # Modify DWT coefficients in the masked region
        coeffs_array[mask] = coeffs_array[mask].mean()  # Example modification
        modified_coeffs = pywt.array_to_coeffs(coeffs_array, coeffs_slices, output_format='wavedec2')
    
        # Reconstruct the modified signal
        reconstructed = pywt.waverec2(modified_coeffs, wavelet='haar', axes=spatial_dims)
        return torch.tensor(reconstructed).to(self.device), mask

        
        # Reconstruct the modified signal
        reconstructed = pywt.waverec2(modified_coeffs, wavelet='haar', axes=spatial_dims)
        return torch.tensor(reconstructed).to(self.device), mask


    def _get_watermarking_mask(self, gt_patch):
        coeffs = pywt.wavedec2(gt_patch.squeeze().cpu().numpy(), wavelet='haar', level=2)
        coeffs_array, _ = pywt.coeffs_to_array(coeffs)
        mask = self._circle_mask(coeffs_array.shape[0], self.w_radius)
        return torch.tensor(mask, dtype=torch.bool).to(self.device)

    def _gen_gt(self, generator=None):
        gt_init = randn_tensor(self.shape, generator=generator, device=self.device, dtype=self.dtype)
        gt_patch, watermarking_mask = self._get_watermarking_pattern(gt_init)
        return gt_patch, watermarking_mask

    def inject_watermark(self, latents):
        coeffs = pywt.wavedec2(latents.squeeze().cpu().numpy(), wavelet='haar', level=2)
        coeffs_array, coeffs_slices = pywt.coeffs_to_array(coeffs)

        coeffs_array[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].cpu().numpy()
        modified_coeffs = pywt.array_to_coeffs(coeffs_array, coeffs_slices, output_format='wavedec2')

        latents_w = pywt.waverec2(modified_coeffs, wavelet='haar')
        return torch.tensor(latents_w).to(self.device)

    def eval_watermark(self, latents_w):
        coeffs = pywt.wavedec2(latents_w.squeeze().cpu().numpy(), wavelet='haar', level=2)
        coeffs_array, _ = pywt.coeffs_to_array(coeffs)

        l1_metric = np.abs(coeffs_array[self.watermarking_mask.cpu().numpy()] - 
                           self.gt_patch[self.watermarking_mask].cpu().numpy()).mean()
        return l1_metric

    def watermark_stat(self):
        dis_all = []
        for i in range(1000):
            rand_latents = randn_tensor(self.shape, device=self.device, dtype=self.dtype)
            dis = self.eval_watermark(rand_latents)
            dis_all.append(dis)
        dis_all = np.array(dis_all)
        return dis_all.mean(), dis_all.var()

    def one_minus_p_value(self, latents):
        l1_metric = self.eval_watermark(latents)
        return abs(0.5 - norm.cdf(l1_metric, self.mu, self.sigma)) * 2

    def tree_ring_p_value(self, latents):
        coeffs = pywt.wavedec2(latents.squeeze().cpu().numpy(), wavelet='haar', level=2)
        coeffs_array, _ = pywt.coeffs_to_array(coeffs)

        target_patch = self.gt_patch[self.watermarking_mask].cpu().numpy().flatten()
        sigma_w = coeffs_array[self.watermarking_mask.cpu().numpy()].std()

        lambda_w = (target_patch ** 2 / sigma_w ** 2).sum()
        x_w = (((coeffs_array[self.watermarking_mask.cpu().numpy()] - target_patch) / sigma_w) ** 2).sum()

        p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)
        return p_w
