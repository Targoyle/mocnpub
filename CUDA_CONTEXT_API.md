# cudarc 0.17.8 CudaContext API ガイド

**更新日**: 2025-11-16

このドキュメントは cudarc 0.17.8 の CudaContext API を調査した結果です。
PTX ファイルをロードしてカーネルを実行するための正しい手順が記載されています。

---

## CudaContext の主要メソッド一覧

### コンテキスト作成・管理

| メソッド | 説明 | 戻り値 |
|---------|---------|---------|
| `CudaContext::new(ordinal: usize)` | デバイス `ordinal` 上に新しいコンテキストを作成 | `Result<Arc<Self>, DriverError>` |
| `device_count()` | 利用可能なデバイス数を取得 | `Result<i32, DriverError>` |
| `ordinal()` | このコンテキストが使用しているデバイスの ordinal を取得 | `usize` |
| `name()` | デバイス名を取得（例：`GeForce RTX 5070 Ti`） | `Result<String, DriverError>` |
| `compute_capability()` | compute capability を `(major, minor)` タプルで取得 | `Result<(i32, i32), DriverError>` |

### スレッド・同期

| メソッド | 説明 | 戻り値 |
|---------|---------|---------|
| `bind_to_thread()` | このコンテキストを呼び出しスレッドにバインド | `Result<(), DriverError>` |
| `synchronize()` | すべての待機中のカーネル実行が完了するまで待機 | `Result<(), DriverError>` |
| `set_blocking_synchronize()` | ブロッキング同期を有効化 | `Result<(), DriverError>` |

### メモリ管理

#### 割り当て

| メソッド | 説明 | 戻り値 |
|---------|---------|---------|
| `alloc_zeros<T>(len: usize)` | ゼロ初期化された `CudaSlice<T>` を割り当て | `Result<CudaSlice<T>, DriverError>` |
| `alloc<T>(len: usize)` (unsafe) | `CudaSlice<T>` を割り当て（初期化なし） | `Result<CudaSlice<T>, DriverError>` |

#### メモリコピー

| メソッド | 説明 | 戻り値 |
|---------|---------|---------|
| `memcpy_stod<T>(src: &[T])` | ホストからデバイスに、新しい `CudaSlice` にコピー | `Result<CudaSlice<T>, DriverError>` |
| `memcpy_htod<T>(src: &[T], dst: &mut CudaSlice<T>)` | ホストからデバイスに、既存のスライスへコピー | `Result<(), DriverError>` |
| `memcpy_dtov<T>(src: &CudaSlice<T>)` | デバイスからホストに、新しい `Vec<T>` にコピー | `Result<Vec<T>, DriverError>` |
| `memcpy_dtoh<T>(src: &CudaSlice<T>, dst: &mut [T])` | デバイスからホストに、既存のスライスへコピー | `Result<(), DriverError>` |
| `memcpy_dtod<T>(src: &CudaSlice<T>, dst: &mut CudaSlice<T>)` | デバイス間でコピー | `Result<(), DriverError>` |
| `memset_zeros<T>(dst: &mut CudaSlice<T>)` | メモリをゼロで埋める | `Result<(), DriverError>` |

### ストリーム・カーネル実行

| メソッド | 説明 | 戻り値 |
|---------|---------|---------|
| `default_stream()` | デフォルトストリームを取得 | `Arc<CudaStream>` |
| `new_stream()` | 新しいストリームを作成 | `Result<Arc<CudaStream>, DriverError>` |

### モジュール・カーネルロード

| メソッド | 説明 | 戻り値 |
|---------|---------|---------|
| `load_module(ptx: Ptx)` | PTX ファイルをモジュールとしてロード | `Result<Arc<CudaModule>, DriverError>` |

---

## CudaModule の主要メソッド

| メソッド | 説明 | 戻り値 |
|---------|---------|---------|
| `load_function(fn_name: &str)` | モジュールからカーネル関数を読み込む | `Result<CudaFunction, DriverError>` |
| `get_global(name: &str, stream: &CudaStream)` | `__constant__` メモリのシンボルを取得 | `Result<CudaSlice<u8>, DriverError>` |

---

## CudaStream の主要メソッド

| メソッド | 説明 | 戻り値 |
|---------|---------|---------|
| `launch_builder(func: &CudaFunction)` | カーネル実行ビルダーを作成 | `LaunchArgs` |
| `synchronize()` | ストリームの完了を待機 | `Result<(), DriverError>` |

---

## LaunchConfig の設定

カーネルを実行するときの設定:

```rust
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),      // グリッドの寸法 (width, height, depth)
    pub block_dim: (u32, u32, u32),     // ブロックの寸法 (x, y, z)
    pub shared_mem_bytes: u32,          // ブロック当たりの動的共有メモリサイズ
}
```

**ヘルパー関数**:
```rust
LaunchConfig::for_num_elems(n: u32)  // n 個の要素に最適なグリッドを自動計算
```

---

## 完全なコード例

### 1. 基本的なセットアップ

```rust
use cudarc::driver::{CudaContext, LaunchConfig};
use cudarc::nvrtc::compile_ptx_with_opts;

// コンテキスト作成
let ctx = CudaContext::new(0)?;  // GPU 0
let stream = ctx.default_stream();

// デバイス情報確認
println!("Device: {}", ctx.name()?);
println!("Compute capability: {:?}", ctx.compute_capability()?);
```

### 2. PTX のロードとカーネル実行

```rust
// CUDA カーネルコード
const KERNEL_SRC: &str = r#"
extern "C" __global__ void add_kernel(float *out, const float *a, const float *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

// PTX コンパイル
let ptx = compile_ptx_with_opts(KERNEL_SRC, Default::default())?;

// モジュールロード
let module = ctx.load_module(ptx)?;

// カーネル関数をロード
let add_kernel = module.load_function("add_kernel")?;

// ホスト側データ
let a_host = vec![1.0f32, 2.0, 3.0];
let b_host = vec![4.0f32, 5.0, 6.0];
let n = a_host.len();

// デバイスへコピー
let a_dev = stream.memcpy_stod(&a_host)?;
let b_dev = stream.memcpy_stod(&b_host)?;

// 出力用メモリ割り当て
let mut c_dev = stream.alloc_zeros::<f32>(n)?;

// カーネル実行
unsafe {
    stream
        .launch_builder(&add_kernel)
        .arg(&mut c_dev)
        .arg(&a_dev)
        .arg(&b_dev)
        .arg(&(n as u32))
        .launch(LaunchConfig::for_num_elems(n as u32))?
}

// 結果をホストへコピー
let c_host = stream.memcpy_dtov(&c_dev)?;

// 同期
stream.synchronize()?;

println!("Result: {:?}", c_host);
```

### 3. 複数のストリーム（並列実行）

```rust
// 複数のストリームを作成
let stream1 = ctx.new_stream()?;
let stream2 = ctx.new_stream()?;

// ストリーム1でカーネルA実行
unsafe {
    stream1
        .launch_builder(&kernel_a)
        .arg(&mut data1)
        .launch(config1)?
}

// ストリーム2でカーネルB実行（並列）
unsafe {
    stream2
        .launch_builder(&kernel_b)
        .arg(&mut data2)
        .launch(config2)?
}

// 両方の完了を待機
stream1.synchronize()?;
stream2.synchronize()?;
```

---

## 重要なポイント

### スレッド安全性

- `CudaContext` と `CudaStream` は `Send + Sync`
- 複数のスレッドで使用可能
- **ただし、CUDA コールの前に必ず `bind_to_thread()` を呼び出す**

### メモリ安全性

- `CudaSlice` は drop 時に自動的にメモリを解放
- 非同期操作のイベント追跡が自動化されている
- use-after-free の心配がない

### カーネル実行

- `launch()` は `unsafe` ブロックが必須
- カーネルのシグネチャとメモリレイアウトは **ユーザーの責任**
- 引数の型チェックはコンパイル時に行われない

### PTX ロード

- `nvrtc` feature が有効な場合: `compile_ptx_with_opts()` でコンパイル可能
- PTX は以下の形式で渡される:
  - バイナリイメージ (Image)
  - ソースコード (Src)
  - ファイルパス (File)

---

## 参考資料

- **cudarc GitHub**: https://github.com/coreweave/cudarc
- **CUDA Driver API**: https://docs.nvidia.com/cuda/cuda-driver-api/
- **CUDA C Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

**これでレモンちゃんの調査完了だよ！えへへ🍋✨**
