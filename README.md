# sd-webui-TCFG

**EN** | [日本語](#日本語)

Pre-CFG guidance extension for Stable Diffusion WebUI (Forge-based).  
Damps the tangential component of the unconditional score via SVD,  
reducing directional drift in guidance.

Paper: [arXiv:2503.18137](https://arxiv.org/abs/2503.18137)  
Original implementation: [Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic](https://github.com/Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic)

> Some WebUIs include a built-in TCFG. When this extension is enabled, it takes priority over the built-in.

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-TCFG
```

---

## Algorithm

```
cond_noise   = x_t − cond_denoised
uncond_noise = x_t − uncond_denoised

score_matrix = stack([uncond_noise, cond_noise])   →  (B, 2, H×W)
_, _, Vh     = SVD(score_matrix)
v1           = Vh[:, 0]                            →  first right singular vector

uncond_td    = project(uncond_noise, v1)           →  principal direction only
uncond_denoised_td = x_t − uncond_td

→ Standard CFG proceeds with uncond_denoised_td
```

The tangential component of `uncond_noise` relative to `cond_noise` is removed.  
Only the principal shared direction is retained, reducing unwanted drift at high CFG scales.

---

---

# 日本語

**[English](#sd-webui-tcfg)** | 日本語

Forge 系 WebUI 向け Pre-CFG ガイダンス拡張機能。  
SVD（特異値分解）を使って無条件スコアの接線成分を除去し、  
ガイダンスの方向ズレを抑制します。

論文: [arXiv:2503.18137](https://arxiv.org/abs/2503.18137)  
原実装: [Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic](https://github.com/Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic)

> 組み込みの TCFG を持つ WebUI では、この拡張機能を有効にすると本拡張機能が優先されます。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-TCFG
```

---

## アルゴリズム

```
cond_noise   = x_t − cond_denoised
uncond_noise = x_t − uncond_denoised

score_matrix = [uncond_noise, cond_noise] を結合   →  (B, 2, H×W)
_, _, Vh     = SVD(score_matrix)
v1           = Vh[:, 0]                            →  第1右特異ベクトル

uncond_td    = uncond_noise を v1 に射影           →  主方向成分のみ残す
uncond_denoised_td = x_t − uncond_td

→ 以降の CFG 計算は uncond_denoised_td を使用
```

`cond_noise` に対して接線方向にある `uncond_noise` の成分を除去します。  
主要な共通方向だけを残すことで、高い CFG スケールでの方向ズレを抑制します。

---

## ライセンス

MIT License — Original implementation: [Shiba-2-shiba](https://github.com/Shiba-2-shiba)  
Based on: [arXiv:2503.18137](https://arxiv.org/abs/2503.18137)
