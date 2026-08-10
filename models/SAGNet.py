import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """Lightweight spatial attention"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


class ShadowAwareIlluminationDecomposition(nn.Module):
    """SAD - Shadow-Aware Detection"""
    def __init__(self, channels=16):
        super().__init__()
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )
        self.illum_extract = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        self.reflect_extract = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shadow_mask    = self.shadow_detector(x)
        illumination   = self.illum_extract(x)
        reflectance    = self.reflect_extract(x)
        reconstruction = reflectance * illumination
        return illumination, reflectance, reconstruction, shadow_mask


# ─────────────────────────────────────────────────────────────────────────────
class MaskCoherenceRefiner(nn.Module):
    """
    Enforces spatial coherence in SACP shadow masks to suppress texture-noise
    false positives (e.g. patterned tile floors activating as 'shadow').
    """
    def __init__(self, smooth_k: int = 9):
        super().__init__()

        # 1. Region smoother: 9x9 conv, initialised as uniform average filter
        self.region_smooth = nn.Conv2d(
            1, 1, smooth_k, padding=smooth_k // 2, groups=1, bias=False
        )
        nn.init.constant_(self.region_smooth.weight, 1.0 / (smooth_k * smooth_k))

        # 3. Blend gate: [smooth_gated_mask, original_mask] -> alpha weight
        self.blend_gate = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=True),
            nn.Sigmoid()
        )
        # Initialise blend to favour coherence path slightly (alpha ~= 0.6)
        nn.init.constant_(self.blend_gate[0].weight, 0.1)
        nn.init.constant_(self.blend_gate[0].bias,   0.4)

    def forward(self, mask: torch.Tensor, x_rgb: torch.Tensor) -> torch.Tensor:
        
        # Step 1: region smoothing
        smooth_mask = torch.clamp(self.region_smooth(mask), 0.0, 1.0)

        # Step 2: luminance-consistency gate (physics-based, no parameters)
        # x_rgb is normalised [-1,1]; convert to [0,1]
        x_01       = x_rgb * 0.5 + 0.5
        luminance  = (0.299 * x_01[:, 0:1] +
                      0.587 * x_01[:, 1:2] +
                      0.114 * x_01[:, 2:3])
        global_lum = luminance.mean(dim=[2, 3], keepdim=True)
        # gate ~= 1 where pixel is darker than global mean (likely shadow)
        # gate ~= 0 where pixel is brighter than global mean (suppress noise)
        lum_gate     = torch.sigmoid(8.0 * (global_lum - luminance))
        smooth_gated = smooth_mask * lum_gate

        # Step 3: learnable blend
        alpha         = self.blend_gate(torch.cat([smooth_gated, mask], dim=1))
        coherent_mask = alpha * smooth_gated + (1.0 - alpha) * mask

        return torch.clamp(coherent_mask, 0.0, 1.0)
# ─────────────────────────────────────────────────────────────────────────────


class ShadowAwareContextPerception(nn.Module):
    """
    SACP - Shadow-Aware Context Perception
    """
    def __init__(self, in_channels=3, channels=24):
        super().__init__()
        branch_ch = channels // 3   # 8 channels per branch

        # Three dilated branches: d=1 (local), d=3 (medium), d=6 (large)
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.medium_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 3, padding=3, dilation=3, bias=False),
            nn.ReLU(inplace=True)
        )
        self.large_branch = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 3, padding=6, dilation=6, bias=False),
            nn.ReLU(inplace=True)
        )

        # Global illumination calibration (SE-style)
        self.global_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Coarse mask head (depthwise-separable)
        self.coarse_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.GroupNorm(4, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # Boundary-aware refinement: image(3) + coarse(1) + initial_SAID(1) = 5ch
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(in_channels + 2, channels // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Mask Coherence Refiner: suppresses texture-noise false positives
        self.mcr = MaskCoherenceRefiner(smooth_k=9)

    def forward(self, x, initial_mask):
        # 1. Multi-scale feature extraction
        multi_scale = torch.cat([
            self.local_branch(x),
            self.medium_branch(x),
            self.large_branch(x)
        ], dim=1)                                            # [B,24,H,W]

        # 2. Global illumination calibration
        multi_scale = multi_scale * self.global_se(x)

        # 3. Coarse shadow mask
        coarse_mask = self.coarse_head(multi_scale)          # [B,1,H,W]

        # 4. Boundary-aware refinement
        refined_mask = self.boundary_refine(
            torch.cat([x, coarse_mask, initial_mask], dim=1)
        )                                                     # [B,1,H,W]

        # 5. Mask Coherence Refiner: suppress texture false positives
        coherent_mask = self.mcr(refined_mask, x)            # [B,1,H,W]

        return coherent_mask, coarse_mask


class ShadowAwareFeatureModulation(nn.Module):
    """Shadow-aware feature modulation using shadow/non-shadow branches"""
    def __init__(self, channels):
        super().__init__()
        self.shadow_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.non_shadow_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x, shadow_mask):
        if shadow_mask.size(2) != x.size(2) or shadow_mask.size(3) != x.size(3):
            shadow_mask = F.interpolate(shadow_mask,
                                        size=(x.size(2), x.size(3)),
                                        mode='bilinear', align_corners=False)
        shadow_feat     = self.shadow_branch(x)     * shadow_mask
        non_shadow_feat = self.non_shadow_branch(x) * (1 - shadow_mask)
        return self.fusion(torch.cat([shadow_feat, non_shadow_feat], dim=1)) + x


class CrossScaleAttentionFusion(nn.Module):
    """CSAF - Cross-Scale Attention Fusion"""
    def __init__(self, channels):
        super().__init__()
        self.scale1 = nn.Conv2d(channels, channels // 2, 3, padding=1,
                                groups=channels // 2, bias=False)
        self.scale2 = nn.Conv2d(channels, channels // 2, 3, padding=2,
                                dilation=2, groups=channels // 2, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = torch.cat([self.scale1(x), self.scale2(x)], dim=1)
        return self.spatial_att(self.fusion(out)) + x


class EfficientConvBlock(nn.Module):
    """Depthwise-separable convolution block"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.norm   = nn.GroupNorm(8, dim)
        self.pwconv = nn.Conv2d(dim, dim, 1, bias=False)
        self.act    = nn.GELU()

    def forward(self, x):
        return self.act(self.pwconv(self.norm(self.dwconv(x)))) + x

# Light Convolutional Block (LCB)
class LightTransformerBlock(nn.Module):
    """Lightweight Convolutional Block (LCB) with CSAF and shadow modulation"""
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.norm1          = nn.GroupNorm(8, dim)
        self.efficient_conv = EfficientConvBlock(dim)
        self.norm2          = nn.GroupNorm(8, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * 2), 1), nn.GELU(),
            nn.Conv2d(int(dim * 2), dim, 1)
        )
        self.csaf              = CrossScaleAttentionFusion(dim)
        self.shadow_modulation = ShadowAwareFeatureModulation(dim)

    def forward(self, x, shadow_mask):
        x = self.shadow_modulation(x, shadow_mask)
        x = x + self.efficient_conv(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.csaf(x)


class UncertaintyGuidedProgressiveRefinement(nn.Module):
    """UGPR - Uncertainty-Guided Progressive Refinement"""
    def __init__(self, channels):
        super().__init__()
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        uncertainty = self.uncertainty_estimator(x)
        refined1    = self.refine1(x) * uncertainty + x * (1 - uncertainty)
        refined2    = self.spatial_att(self.refine2(refined1))
        refined2    = refined2 * uncertainty + refined1 * (1 - uncertainty)
        return refined2, uncertainty


class SAGNet(nn.Module):
    """
    SAGNet: Shadow-Aware Guided restoration Networt
            -Shadow-Aware: internally self-detected shadow mask
            -Guided: self-detected mask guides the entire LC-block pathway via SAFM
            -The self-detected shadow mask internally guides the entire restoration pathway

      1. SAD  - Shadow-Aware Detection
      2. CSAF  - Cross-Scale Attention Fusion
      3. UGPR  - Uncertainty-Guided Progressive Refinement
      4. SACP  - Shadow-Aware Context Perception
    """
    def __init__(self,
                 input_channels=3,
                 output_channels=3,
                 embed_dim=64,
                 num_blocks=[2, 2, 2, 2],
                 num_heads=[2, 4, 8, 16]):
        super().__init__()

        self.said = ShadowAwareIlluminationDecomposition(channels=16)
        self.sacp = ShadowAwareContextPerception(in_channels=3, channels=24)

        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]

        # Encoder
        self.downsample_layers = nn.ModuleList()
        self.encoder_stages    = nn.ModuleList()
        for i in range(4):
            ds = nn.Identity() if i == 0 else nn.Sequential(
                nn.Conv2d(dims[i-1], dims[i], 2, stride=2, bias=False),
                nn.ReLU(inplace=True)
            )
            self.downsample_layers.append(ds)
            self.encoder_stages.append(nn.ModuleList([
                LightTransformerBlock(dims[i], num_heads[i])
                for _ in range(num_blocks[i])
            ]))

        # Bottleneck LCB
        self.bottleneck = nn.Sequential(
            LightTransformerBlock(dims[3], num_heads[3]),
            LightTransformerBlock(dims[3], num_heads[3])
        )

        # Decoder
        self.decoder_stages   = nn.ModuleList()
        self.upsample_layers  = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        for i in range(3, -1, -1):
            self.skip_connections.append(
                nn.Conv2d(dims[i] * 2, dims[i], 1, bias=False)
                if i < 3 else nn.Identity()
            )
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dims[i], dims[i-1], 2, stride=2, bias=False),
                    nn.ReLU(inplace=True)
                ) if i > 0 else nn.Identity()
            )
            self.decoder_stages.append(nn.ModuleList([
                LightTransformerBlock(dims[i], num_heads[i])
                for _ in range(num_blocks[i])
            ]))

        self.ugpr = UncertaintyGuidedProgressiveRefinement(dims[0])

        self.output_proj = nn.Sequential(
            nn.Conv2d(dims[0], dims[0]//2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0]//2, output_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # SAID: illumination decomposition
        illumination, reflectance, recon, initial_mask = self.said(x)

        # SACP + MCR: coherent shadow mask (texture noise suppressed)
        shadow_mask, coarse_mask = self.sacp(x, initial_mask)

        feat = self.input_proj(x)

        # Encoder
        encoder_features = []
        for ds, blocks in zip(self.downsample_layers, self.encoder_stages):
            feat = ds(feat)
            for blk in blocks:
                feat = blk(feat, shadow_mask)
            encoder_features.append(feat)

        # Bottleneck
        for layer in self.bottleneck:
            feat = layer(feat, shadow_mask)

        # Decoder
        for blk in self.decoder_stages[0]:
            feat = blk(feat, shadow_mask)
        feat = self.upsample_layers[0](feat)

        feat = self.skip_connections[1](torch.cat([feat, encoder_features[2]], dim=1))
        for blk in self.decoder_stages[1]:
            feat = blk(feat, shadow_mask)
        feat = self.upsample_layers[1](feat)

        feat = self.skip_connections[2](torch.cat([feat, encoder_features[1]], dim=1))
        for blk in self.decoder_stages[2]:
            feat = blk(feat, shadow_mask)
        feat = self.upsample_layers[2](feat)

        feat = self.skip_connections[3](torch.cat([feat, encoder_features[0]], dim=1))
        for blk in self.decoder_stages[3]:
            feat = blk(feat, shadow_mask)
        feat = self.upsample_layers[3](feat)

        # UGPR
        feat, uncertainty = self.ugpr(feat)

        final_out = self.output_proj(feat)

        return {
            'output':         final_out,
            'illumination':   illumination,
            'reflectance':    reflectance,
            'reconstruction': recon,
            'shadow_mask':    shadow_mask,   # coherent SACP+MCR mask
            'coarse_mask':    coarse_mask,   # SACP pre-MCR (for aux loss)
            'initial_mask':   initial_mask,  # raw SAID mask
            'uncertainty':    uncertainty
        }

    def test_set(self, x):
        return self.forward(x)['output']


if __name__ == '__main__':
    model  = SAGNet(input_channels=3, output_channels=3, embed_dim=64,
                       num_blocks=[2, 2, 2, 2], num_heads=[2, 4, 8, 16])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = model.to(device)
    x      = torch.randn(2, 3, 256, 256).to(device)

    with torch.no_grad():
        out = model(x)

    print(f"Output       : {out['output'].shape}")
    print(f"Shadow mask  : {out['shadow_mask'].shape}  <- SACP + MCR (coherent)")
    print(f"Coarse mask  : {out['coarse_mask'].shape}  <- SACP pre-MCR")
    print(f"Initial mask : {out['initial_mask'].shape} <- raw SAID")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {total:,}  ({total/1e6:.4f} M)")