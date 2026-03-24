import io
import cv2
import lpips
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL
from scipy.ndimage import gaussian_filter
from spandrel import ModelLoader


class YCrCbResidualBuilder:
    def __init__(
        self,
        use_jpeg=True,
        use_blur=True,
        use_vae=True,
        use_sr=True,
        jpeg_quality=75,
        blur_sigma=2.0,
        vae_model_id="stabilityai/sd-vae-ft-ema", # https://cs231n.stanford.edu/2025/papers/text_file_840591073-CS_231N_Project_Report.pdf
        realesrgan_model_path="models\\RealESRGAN_x2plus.pth", #https://github.com/xinntao/Real-ESRGAN
        device="cpu",
        vae_size=(512, 512),
    ):
        self.use_jpeg = use_jpeg
        self.use_blur = use_blur
        self.use_vae = use_vae
        self.use_sr = use_sr

        self.jpeg_quality = jpeg_quality
        self.blur_sigma = blur_sigma
        self.vae_model_id = vae_model_id
        self.realesrgan_model_path = realesrgan_model_path
        self.device = device
        self.vae_size = vae_size

        self.vae = None
        self.lpips_fn = None
        self.sr_model = None

        if self.use_vae or self.use_sr:
            self._load_vae_models()

        if self.use_sr:
            self._load_sr_models()

        self.transform = T.Compose([
            T.Resize(self.vae_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def extract(self, image_rgb):

        image_u8 = self._to_uint8_rgb(image_rgb)
        outputs = {}

        if self.use_jpeg:
            outputs["jpeg_residual"] = self._jpeg_residual(image_u8)

        if self.use_blur:
            outputs["blur_residual"] = self._blur_residual(image_u8)

        if self.use_vae:
            outputs["vae_residual"] = self._vae_residual(image_u8)
            outputs["vae_lpips"] = self._vae_lpips(image_u8)

        if self.use_sr:
            outputs["sr_residual"] = self._sr_residual(image_u8)
            outputs["sr_delta_re"] = self._sr_delta(image_u8)

        return outputs

    def extract_with_recons(self, image_rgb):
        image_u8 = self._to_uint8_rgb(image_rgb)
        outputs = {}

        if self.use_jpeg:
            recon = self._jpeg_recon(image_u8)
            outputs["jpeg"] = {
                "reconstruction_rgb": recon,
                "reconstruction_ycrcb": self._rgb_to_ycrcb(recon),
                "residual_ycrcb": self._compute_ycrcb_residual(image_u8, recon),
            }

        if self.use_blur:
            recon = self._blur_recon(image_u8)
            outputs["blur"] = {
                "reconstruction_rgb": recon,
                "reconstruction_ycrcb": self._rgb_to_ycrcb(recon),
                "residual_ycrcb": self._compute_ycrcb_residual(image_u8, recon),
            }

        if self.use_vae:
            recon = self._vae_recon(image_u8)
            outputs["vae"] = {
                "reconstruction_rgb": recon,
                "reconstruction_ycrcb": self._rgb_to_ycrcb(recon),
                "residual_ycrcb": self._compute_ycrcb_residual(image_u8, recon),
                "lpips": self._vae_lpips(image_u8),
            }

        if self.use_sr:
            recon = self._sr_upsample(image_u8)
            outputs["sr"] = {
                "reconstruction_rgb": recon,
                "reconstruction_ycrcb": self._rgb_to_ycrcb(recon),
                "residual_ycrcb": self._compute_ycrcb_residual(image_u8, recon),
                "delta_re": self._sr_delta(image_u8),
            }

        return outputs

    def _jpeg_recon(self, image_u8):
        buf = io.BytesIO()
        Image.fromarray(image_u8).save(buf, format="JPEG", quality=self.jpeg_quality)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))

    def _jpeg_residual(self, image_u8):
        recon = self._jpeg_recon(image_u8)
        return self._compute_ycrcb_residual(image_u8, recon)

    def _blur_recon(self, image_u8):
        image_f = image_u8.astype(np.float32)
        blurred = cv2.GaussianBlur(
            image_f,
            ksize=(0, 0),
            sigmaX=self.blur_sigma,
            sigmaY=self.blur_sigma,
        )
        return np.clip(blurred, 0, 255).round().astype(np.uint8)

    def _blur_residual(self, image_u8):
        recon = self._blur_recon(image_u8)
        return self._compute_ycrcb_residual(image_u8, recon)

    def _load_vae_models(self):
        self.vae = AutoencoderKL.from_pretrained(self.vae_model_id, local_files_only=True).eval().to(self.device)
        self.lpips_fn = lpips.LPIPS(net="vgg").eval().to(self.device)

    def _vae_recon(self, image_u8):
        h, w = image_u8.shape[:2]
        x = self._to_vae_tensor(image_u8)

        with torch.no_grad():
            z = self.vae.encode(x).latent_dist.sample()
            x_recon = self.vae.decode(z).sample

        recon = self._tensor_to_uint8_rgb(x_recon)
        recon = cv2.resize(recon, (w, h), interpolation=cv2.INTER_AREA)
        return recon

    def _vae_residual(self, image_u8):
        recon = self._vae_recon(image_u8)
        return self._compute_ycrcb_residual(image_u8, recon)

    def _vae_lpips(self, image_u8):
        x = self._to_vae_tensor(image_u8)
        with torch.no_grad():
            z = self.vae.encode(x).latent_dist.sample()
            x_recon = self.vae.decode(z).sample
        return float(self.lpips_fn(x, x_recon).item())

    def _load_sr_models(self):
        self.sr_model = ModelLoader().load_from_file(self.realesrgan_model_path).eval()
        if hasattr(self.sr_model, "to"):
            self.sr_model = self.sr_model.to(self.device)

    def _sr_upsample(self, image_u8):
        h, w = image_u8.shape[:2]

        tensor = torch.from_numpy(image_u8).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            sr_out = self.sr_model(tensor).squeeze(0)

        sr_np = (
            sr_out.detach()
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy() * 255.0
        ).round().astype(np.uint8)

        recon = cv2.resize(sr_np, (w, h), interpolation=cv2.INTER_AREA)
        return recon

    def _sr_residual(self, image_u8):
        recon = self._sr_upsample(image_u8)
        return self._compute_ycrcb_residual(image_u8, recon)

    def _sr_delta(self, image_u8):
        re_before = self._vae_lpips(image_u8)
        recon = self._sr_upsample(image_u8)
        re_after = self._vae_lpips(recon)
        return float(re_after - re_before)

    def _compute_ycrcb_residual(self, orig_rgb, recon_rgb):
        orig_ycrcb = self._rgb_to_ycrcb(orig_rgb)
        recon_ycrcb = self._rgb_to_ycrcb(recon_rgb)
        residual = np.abs(orig_ycrcb - recon_ycrcb)
        return residual.astype(np.float32)

    def _rgb_to_ycrcb(self, image_rgb):
        img = self._to_uint8_rgb(image_rgb)
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)

    def _to_uint8_rgb(self, image):
        img = np.asarray(image)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB (H,W,3), got {img.shape}")

        if img.dtype == np.uint8:
            return img

        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0

        return np.clip(img, 0, 255).round().astype(np.uint8)

    def _to_vae_tensor(self, image_u8):
        return self.transform(Image.fromarray(image_u8)).unsqueeze(0).to(self.device)

    def _tensor_to_uint8_rgb(self, tensor):
        arr = tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        arr = (arr + 1.0) * 127.5
        return np.clip(arr, 0, 255).round().astype(np.uint8)