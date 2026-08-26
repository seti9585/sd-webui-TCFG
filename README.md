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

## Behaviour in a multi-extension pre-CFG chain

Forge-based backends allow several extensions to register a pre-CFG hook. Each one rewrites `conds_out` in place and passes it on, so the order in which those hooks run changes the result and must not be left to chance.

This extension exposes a `_sd_webui_priority` attribute and inserts its hook at priority 13.0 inside the pre-CFG queue, rather than appending it in extension load order. Without this, execution order follows alphabetical load order and shifts depending on which extensions happen to be installed. Other extensions in the same suite that expose the attribute participate in the same ordering.

Note that this is separate from `sorting_priority`, which controls only the display order of the UI accordion and has no effect on callback execution order.

TCFG holds the lowest priority value among the pre-CFG extensions in this suite, so it runs first and receives the raw unconditional prediction. This matches the assumption the algorithm is built on: the tangential projection is taken against an uncond that no other extension has already reshaped.

### Priority values in this suite

| Priority | Extension | Stage |
| --- | --- | --- |
| 13.0 | sd-webui-TCFG | pre-CFG |
| 14.0 | sd-webui-SkimmedCFG | pre-CFG |
| 14.2 | sd-webui-DifferenceCFG | pre-CFG |
| 14.5 | sd-webui-APG | pre-CFG |
| 15.0 | sd-webui-CFGZeroStar | post-CFG |
| 15.2 | sd-webui-FreSca | post-CFG |
| 15.5 | sd-webui-MaHiRo | post-CFG |
| 16.0 | sd-webui-CFGNorm | post-CFG |
| 16.5 | sd-webui-CFGRegulator | post-CFG |

Lower values run earlier. The two stages are separate queues; a pre-CFG hook always runs before any post-CFG hook regardless of the numbers.

---

## Debug output

Set the environment variable `SD_WEBUI_SETI_DEBUG` before launching:

| Value | Effect |
| --- | --- |
| 0 (or unset) | No debug output |
| 1 | Registration logging and the pre-CFG chain dump |

The registration log reports the resolved priority. The authoritative record of actual call order is the chain dump this extension emits at sampling time, which lists every registered pre-CFG hook in the order it runs:

```
[TCFG] pre-CFG chain: _tcfg_pre_cfg_fn(13.0) -> _differencecfg_pre_cfg_fn(14.2)
```

If a hook you expected is missing from the dump, that extension is not enabled or did not register. If the order differs from the table above, an extension in the chain is not participating in priority insertion.

Output goes to both the module logger and stderr, because some backends suppress module-level logger output.

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

## 複数拡張機能が同時に動作する場合の挙動

Forge 系バックエンドでは複数の拡張機能が pre-CFG フックを登録できます。各フックは `conds_out` をその場で書き換えて次へ渡すため、実行順序は結果を変えます。偶然に委ねてはなりません。

本拡張機能は `_sd_webui_priority` 属性を持ち、拡張機能のロード順で末尾に追加されるのではなく、pre-CFG キュー内の優先度 13.0 の位置にフックを挿入します。これを行わない場合、実行順序はアルファベット順のロード順に従い、インストールされている拡張機能の構成によって変動します。同属性を持つ同シリーズの他の拡張機能も、同じ順序付けに参加します。

これは `sorting_priority` とは別物である点に注意してください。`sorting_priority` は UI アコーディオンの表示順のみを制御し、コールバックの実行順序には影響しません。

TCFG は本シリーズの pre-CFG 拡張機能の中で最も小さい優先度値を持つため最初に実行され、加工されていない無条件予測を受け取ります。これはアルゴリズムが前提としている条件と一致します。接線方向の射影は、他の拡張機能によって書き換えられていない uncond に対して行われます。

### 本シリーズの優先度値

| 優先度 | 拡張機能 | 段 |
| --- | --- | --- |
| 13.0 | sd-webui-TCFG | pre-CFG |
| 14.0 | sd-webui-SkimmedCFG | pre-CFG |
| 14.2 | sd-webui-DifferenceCFG | pre-CFG |
| 14.5 | sd-webui-APG | pre-CFG |
| 15.0 | sd-webui-CFGZeroStar | post-CFG |
| 15.2 | sd-webui-FreSca | post-CFG |
| 15.5 | sd-webui-MaHiRo | post-CFG |
| 16.0 | sd-webui-CFGNorm | post-CFG |
| 16.5 | sd-webui-CFGRegulator | post-CFG |

値が小さいほど先に実行されます。2 つの段は別々のキューであり、数値にかかわらず pre-CFG フックは必ず post-CFG フックより先に実行されます。

---

## デバッグ出力

起動前に環境変数 `SD_WEBUI_SETI_DEBUG` を設定します。

| 値 | 動作 |
| --- | --- |
| 0（または未設定） | デバッグ出力なし |
| 1 | 登録時のログと pre-CFG チェーンダンプ |

登録時のログには解決された優先度が出力されます。実際の呼び出し順序を示す正式な記録は、本拡張機能がサンプリング時に出力するチェーンダンプです。登録済みの全 pre-CFG フックが実行順に列挙されます。

```
[TCFG] pre-CFG chain: _tcfg_pre_cfg_fn(13.0) -> _differencecfg_pre_cfg_fn(14.2)
```

想定していたフックがダンプに現れない場合、その拡張機能が有効化されていないか、登録に失敗しています。順序が上表と異なる場合、チェーン内のいずれかの拡張機能が優先度挿入に参加していません。

出力はモジュールロガーと stderr の両方に送られます。一部のバックエンドがモジュールレベルのロガー出力を抑制するためです。

---

## ライセンス

MIT License — Original implementation: [Shiba-2-shiba](https://github.com/Shiba-2-shiba)  
Based on: [arXiv:2503.18137](https://arxiv.org/abs/2503.18137)
