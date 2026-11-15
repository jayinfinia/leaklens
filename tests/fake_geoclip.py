"""A tiny stand-in for GeoCLIP with the same attributes LeakLens uses
(image_encoder.preprocess_image, image_encoder, location_encoder,
logit_scale, gps_gallery). Its image <-> GPS similarity falls off with
distance from `target`, so the right answer is known."""

import math

import torch
from torch import nn


class _ImageEncoder(nn.Module):
    def preprocess_image(self, image):
        return torch.zeros(1, 3, 8, 8)

    def forward(self, x):
        return torch.tensor([[1.0, 0.0, 0.0]]).repeat(x.shape[0], 1)


class _LocationEncoder(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.lat, self.lon = target

    def forward(self, coords):
        lat = torch.deg2rad(coords[:, 0])
        lon = torch.deg2rad(coords[:, 1])
        p, q = math.radians(self.lat), math.radians(self.lon)
        a = torch.sin((lat - p) / 2) ** 2 + math.cos(p) * torch.cos(lat) * torch.sin((lon - q) / 2) ** 2
        km = 2 * 6371 * torch.asin(torch.sqrt(a.clamp(0, 1)))
        return torch.stack([torch.ones_like(km), -km / 300, torch.zeros_like(km)], dim=1)


def us_grid(step: float = 0.5) -> torch.Tensor:
    pts = [(la * step, lo * step) for la in range(int(25 / step), int(49 / step))
           for lo in range(int(-124 / step), int(-67 / step))]
    return torch.tensor(pts, dtype=torch.float32)


class FakeGeoCLIP(nn.Module):
    def __init__(self, target=(34.7304, -86.5861), gallery=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(50.0)))
        self.image_encoder = _ImageEncoder()
        self.location_encoder = _LocationEncoder(target)
        if gallery is None:  # a coarse US grid plus a dense patch around the target, like
            # the real gallery's dense coverage of well-photographed cities
            patch = torch.tensor([(target[0] + i * 0.03, target[1] + j * 0.03)
                                  for i in range(-5, 6) for j in range(-5, 6)], dtype=torch.float32)
            gallery = torch.cat([us_grid(), patch])
        self.gps_gallery = gallery
