# mocnpub タスクリスト 📋

**作成日**: 2025-11-14
**最終更新**: 2025-11-29
**進捗**: Step 0〜4 完了！🎉 Phase 3 最適化を計画中

---

## 📊 全体サマリー

| Step | 概要 | 状態 |
|------|------|------|
| Step 0 | Rust + CUDA の Hello World | ✅ 完了 |
| Step 1 | GPU で簡単なプログラム（マンデルブロ集合）| ✅ 完了 |
| Step 2 | CPU 版 npub マイニング | ✅ 完了 |
| Step 2.5 | CPU 版のブラッシュアップ | ✅ 完了 |
| Step 3 | GPU 版に移行（16倍高速化）| ✅ 完了 |
| Step 4 | GPU カーネル高速化（116,000倍！）| ✅ 完了 |

---

## 🏆 現在のパフォーマンス

| 段階 | スループット | CPU比 |
|------|-------------|-------|
| CPU（16スレッド） | ~70,000 keys/sec | 1x |
| GPU Montgomery 1億連（ベンチ） | **8.1B keys/sec** | **116,000x** 🔥 |
| **実際のマイニング（Windows）** | **~391M keys/sec** | **5,586x** |

**8文字 prefix が約2分で見つかる！** 🎉

---

## ✅ 完了した Step

### Step 0: Rust + CUDA の Hello World 🌸
- [x] CUDA Toolkit 13.0 インストール（Windows + WSL）
- [x] Rust + cudarc 0.17.8 でセットアップ
- [x] RTX 5070 Ti への接続確認

### Step 1: GPU で簡単なプログラム 🔥
- [x] マンデルブロ集合を CPU/GPU で実装
- [x] GPU 版で 3.5倍高速化を達成
- [x] CUDA カーネル、PTX、cudarc の基本を習得

### Step 2: CPU 版 npub マイニング 💪
- [x] secp256k1 と Nostr の鍵生成を理解
- [x] bech32 エンコーディング（npub/nsec）を実装
- [x] CLI インターフェース（clap）を実装
- [x] パフォーマンス測定（~70,000 keys/sec）

### Step 2.5: CPU 版のブラッシュアップ 🔧
- [x] マルチスレッド対応（16スレッド、12〜20倍高速化）
- [x] 入力検証（bech32 無効文字チェック）
- [x] テストコード（7つのテストケース）
- [x] 継続モード（`--limit` オプション）
- [x] 複数 prefix の OR 指定
- [x] ベンチマーク（criterion 0.6）

### Step 3: GPU 版に移行 🚀
- [x] 参考実装の調査（VanitySearch, CudaBrainSecp）
- [x] 独自の secp256k1 CUDA カーネルを実装
- [x] Point Multiplication、バッチ処理を実装
- [x] GPU 版で 16倍高速化を達成

### Step 4: GPU カーネル高速化 🔥🔥🔥
- [x] 連続秘密鍵戦略（1億連ガチャ）
- [x] Montgomery's Trick（逆元のバッチ計算）
- [x] Mixed Addition（G の Z=1 を活用）
- [x] GPU 側 prefix マッチング（bech32 スキップ）

---

## 🚀 将来の最適化計画（Phase 3）

| # | 最適化 | 期待効果 | 優先度 |
|---|--------|----------|--------|
| 1 | **エンドモルフィズム** | **6倍** 🔥 | **最優先** |
| 2 | `_ModSquare` 最適化 | 3-5% | 高 |
| 3 | ncu プロファイリング | 計測 | 高 |
| 4 | パラメータ検証 | 不明 | 中 |

### エンドモルフィズム（β, β²）

secp256k1 の特殊な性質を利用：
- 1回の公開鍵計算で **6つの公開鍵をチェック**
- X座標に β（立方根）を掛けるだけで新しい公開鍵が得られる
- 理論上 **6倍の高速化** 🚀

### 見送った最適化

| 最適化 | 理由 |
|--------|------|
| 2^i × G プリコンピュート | 効果 0.2%（`_PointMult` が全体の 0.4%）|
| PTX carry/borrow | NVCC が最適化済み |
| CUDA Streams | 転送がボトルネックではない |

---

## 📝 備考

**作業履歴の詳細**：`~/sakura-chan/diary/` と `~/sakura-chan/work/` に記録

**技術的な詳細**：`CLAUDE.md` を参照
