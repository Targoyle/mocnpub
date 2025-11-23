use clap::Parser;
use secp256k1::rand;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use bech32::{encode, Bech32, Hrp};
use hex;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::time::Instant;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

/// Nostr npub マイニングツール 🔑
///
/// 指定した prefix を持つ npub（Nostr 公開鍵）を見つけるマイニングツール。
/// CPU 版の実装で、GPU 版は Step 3 で実装予定。
#[derive(Parser, Debug)]
#[command(name = "mocnpub")]
#[command(about = "Nostr npub マイニングツール 🔑", long_about = None)]
struct Args {
    /// マイニングする prefix（npub1 に続く bech32 文字列）
    ///
    /// 単一 prefix: "abc", "test", "satoshi"
    /// 複数 prefix（OR 指定）: "m0ctane0,m0ctane2,m0ctane3"（カンマ区切り）
    /// 完全な npub 例: npub1abc... の "abc" 部分を指定
    #[arg(short, long)]
    prefix: String,

    /// 結果を出力するファイル（オプション、デフォルトは stdout）
    #[arg(short, long)]
    output: Option<String>,

    /// スレッド数（デフォルト: CPU コア数を自動検出）
    #[arg(short, long)]
    threads: Option<usize>,

    /// 見つける鍵の個数（0 = 無限、デフォルト: 1）
    #[arg(short, long, default_value = "1")]
    limit: usize,
}

/// 公開鍵（x座標のみ32バイト）を npub に変換
fn pubkey_to_npub(pubkey: &PublicKey) -> String {
    // 公開鍵の hex 文字列を取得（圧縮形式）
    let pk_hex = pubkey.to_string();
    // x座標のみを抽出（先頭2文字を除去）
    let pk_x_only = &pk_hex[2..];

    // hex 文字列を 32 バイトのバイト列に変換
    let mut bytes = [0u8; 32];
    hex::decode_to_slice(pk_x_only, &mut bytes).expect("Invalid hex string");

    // bech32 エンコード
    let hrp = Hrp::parse("npub").expect("valid hrp");
    encode::<Bech32>(hrp, &bytes).expect("failed to encode npub")
}

/// 秘密鍵（32バイト）を nsec に変換
fn seckey_to_nsec(seckey: &SecretKey) -> String {
    // 秘密鍵のバイト列を取得
    let bytes = seckey.secret_bytes();

    // bech32 エンコード
    let hrp = Hrp::parse("nsec").expect("valid hrp");
    encode::<Bech32>(hrp, &bytes).expect("failed to encode nsec")
}

/// prefix の妥当性を検証（bech32 の有効文字のみを許可）
///
/// bech32 で使用可能な文字: 023456789acdefghjklmnpqrstuvwxyz (32文字)
/// 使用不可な文字: 1, b, i, o（混同を避けるため除外されている）
///
/// # Returns
/// - Ok(()) : prefix が有効
/// - Err(String) : エラーメッセージ
fn validate_prefix(prefix: &str) -> Result<(), String> {
    // bech32 の有効な文字セット（32文字）
    const VALID_CHARS: &str = "023456789acdefghjklmnpqrstuvwxyz";

    // 空文字チェック
    if prefix.is_empty() {
        return Err("Prefix cannot be empty".to_string());
    }

    // 各文字をチェック
    for (i, ch) in prefix.chars().enumerate() {
        // 大文字をチェック
        if ch.is_uppercase() {
            return Err(format!(
                "Invalid prefix '{}': bech32 does not allow uppercase letters (found '{}' at position {})\n\
                 Hint: Use lowercase instead",
                prefix, ch, i
            ));
        }

        // bech32 で無効な文字をチェック
        if !VALID_CHARS.contains(ch) {
            // 特に混同しやすい文字には詳しい説明を追加
            let hint = match ch {
                '1' => "Character '1' is not allowed (reserved as separator in bech32)",
                'b' | 'i' | 'o' => "Character is excluded to avoid confusion with similar-looking characters",
                _ => "Character is not in the bech32 character set",
            };

            return Err(format!(
                "Invalid prefix '{}': bech32 does not allow '{}'\n\
                 {}\n\
                 Valid characters: {}",
                prefix, ch, hint, VALID_CHARS
            ));
        }
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    // prefix をカンマ区切りで split して Vec に変換
    let prefixes: Vec<String> = args.prefix
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // 各 prefix の妥当性を検証
    for prefix in &prefixes {
        if let Err(err_msg) = validate_prefix(prefix) {
            eprintln!("❌ Error: {}", err_msg);
            std::process::exit(1);
        }
    }

    // スレッド数を決定（引数指定 or CPU コア数）
    let num_threads = args.threads.unwrap_or_else(num_cpus::get);

    println!("🔥 mocnpub - Nostr npub マイニング 🔥");
    if prefixes.len() == 1 {
        println!("Prefix: '{}'", prefixes[0]);
    } else {
        println!("Prefixes (OR): {}", prefixes.join(", "));
    }
    println!("Threads: {}", num_threads);
    println!("Limit: {}\n", if args.limit == 0 { "無限".to_string() } else { args.limit.to_string() });

    // 全スレッド共有のカウンタ
    let total_count = Arc::new(AtomicU64::new(0));
    let found_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    // prefixes を Arc で共有
    let prefixes = Arc::new(prefixes);

    // channel を作成（ワーカースレッド → メインスレッド）
    // (SecretKey, PublicKey, npub, matched_prefix, 試行回数)
    let (sender, receiver) = mpsc::channel::<(SecretKey, PublicKey, String, String, u64)>();

    // スレッドを起動
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let prefixes = Arc::clone(&prefixes);
            let total_count = Arc::clone(&total_count);
            let found_count = Arc::clone(&found_count);
            let sender = sender.clone();
            let limit = args.limit;

            std::thread::spawn(move || {
                let secp = Secp256k1::new();
                let mut local_count = 0u64;

                loop {
                    // limit 個見つかったらループを抜ける（0 = 無限の場合は抜けない）
                    if limit > 0 && found_count.load(Ordering::Relaxed) >= limit {
                        break;
                    }

                    let (sk, pk) = secp.generate_keypair(&mut rand::thread_rng());
                    local_count += 1;

                    // bech32 形式に変換
                    let npub = pubkey_to_npub(&pk);
                    // "npub1" を除去して、bech32 文字列の部分だけを取り出す
                    let npub_body = &npub[5..]; // "npub1" は5文字

                    // 複数 prefix のマッチング判定（どれか1つにマッチすれば OK）
                    if let Some(matched_prefix) = prefixes.iter().find(|p| npub_body.starts_with(p.as_str())) {
                        // 見つかった個数をインクリメント
                        let count = found_count.fetch_add(1, Ordering::Relaxed) + 1;

                        // 現在の試行回数を取得
                        let current_total = total_count.load(Ordering::Relaxed) + local_count;

                        // 結果を channel 経由で送信（matched_prefix も含める）
                        if sender.send((sk, pk, npub.clone(), matched_prefix.clone(), current_total)).is_err() {
                            // メインスレッドが終了している場合
                            break;
                        }

                        // limit 個見つかったらループを抜ける（0 = 無限の場合は抜けない）
                        if limit > 0 && count >= limit {
                            break;
                        }
                    }

                    // 定期的に全体カウンタを更新（100回ごと）
                    if local_count % 100 == 0 {
                        total_count.fetch_add(100, Ordering::Relaxed);
                    }
                }

                // 最後に残りのカウントを加算
                let remainder = local_count % 100;
                if remainder > 0 {
                    total_count.fetch_add(remainder, Ordering::Relaxed);
                }
            })
        })
        .collect();

    // sender を drop（全ワーカースレッドが終了したら receiver が None を返すようにする）
    drop(sender);

    // 進捗表示スレッド
    let total_count_progress = Arc::clone(&total_count);
    let found_count_progress = Arc::clone(&found_count);
    let limit_progress = args.limit;
    let progress_handle = std::thread::spawn(move || {
        loop {
            // limit 個見つかったら終了（0 = 無限の場合は終了しない）
            if limit_progress > 0 && found_count_progress.load(Ordering::Relaxed) >= limit_progress {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
            let count = total_count_progress.load(Ordering::Relaxed);
            let found = found_count_progress.load(Ordering::Relaxed);
            if count > 0 {
                println!("{}回試行中... (見つかった: {}個)", count, found);
            }
        }
    });

    // ファイル出力の準備（append モード）
    let mut output_file = if let Some(ref output_path) = args.output {
        Some(OpenOptions::new()
            .create(true)
            .append(true)
            .open(output_path)?)
    } else {
        None
    };

    // メインスレッドで結果を受信・出力
    let mut result_count = 0;
    while let Ok((sk, pk, npub, matched_prefix, current_total)) = receiver.recv() {
        result_count += 1;
        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        let keys_per_sec = current_total as f64 / elapsed_secs;

        let nsec = seckey_to_nsec(&sk);
        let pk_hex = pk.to_string();
        let pk_x_only = &pk_hex[2..]; // x座標のみ（圧縮形式の先頭2文字を除去）

        // 結果を整形
        let output_text = format!(
            "✅ {}個目が見つかりました！（{}回試行、{}スレッド）\n\
             マッチした prefix: '{}'\n\n\
             経過時間: {:.2}秒\n\
             パフォーマンス: {:.2} keys/sec\n\n\
             秘密鍵（hex）: {}\n\
             秘密鍵（nsec）: {}\n\
             公開鍵（圧縮形式）: {}\n\
             公開鍵（x座標のみ）: {}\n\
             公開鍵（npub）: {}\n\
{}\n",
            result_count,
            current_total,
            num_threads,
            matched_prefix,
            elapsed_secs,
            keys_per_sec,
            sk.display_secret(),
            nsec,
            pk,
            pk_x_only,
            npub,
            "=".repeat(80)
        );

        // 出力先に応じて出力
        if let Some(ref mut file) = output_file {
            // ファイルに append
            file.write_all(output_text.as_bytes())?;
            file.flush()?;
        }
        // stdout にも出力（ファイル出力の有無に関わらず）
        print!("{}", output_text);
        io::stdout().flush()?;
    }

    // 全スレッドの終了を待つ
    for handle in handles {
        handle.join().unwrap();
    }
    progress_handle.join().unwrap();

    // 最終結果を表示
    let final_count = total_count.load(Ordering::Relaxed);
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    println!("\n🎉 マイニング完了！");
    println!("見つかった鍵: {}個", result_count);
    println!("総試行回数: {}回", final_count);
    println!("経過時間: {:.2}秒", elapsed_secs);
    if let Some(ref output_path) = args.output {
        println!("結果をファイルに保存しました: {}", output_path);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use secp256k1::SecretKey;

    #[test]
    fn test_validate_prefix_valid() {
        // 有効な prefix のテスト
        assert!(validate_prefix("test").is_ok());
        assert!(validate_prefix("0").is_ok());
        assert!(validate_prefix("00").is_ok());
        assert!(validate_prefix("ac").is_ok());
        assert!(validate_prefix("m0ctane").is_ok());
    }

    #[test]
    fn test_validate_prefix_invalid_chars() {
        // 無効な文字（1, b, i, o）を含む prefix
        assert!(validate_prefix("abc").is_err()); // 'b' が無効
        assert!(validate_prefix("test1").is_err()); // '1' が無効
        assert!(validate_prefix("testi").is_err()); // 'i' が無効
        assert!(validate_prefix("testo").is_err()); // 'o' が無効
    }

    #[test]
    fn test_validate_prefix_uppercase() {
        // 大文字を含む prefix
        assert!(validate_prefix("Test").is_err());
        assert!(validate_prefix("TEST").is_err());
        assert!(validate_prefix("TeSt").is_err());
    }

    #[test]
    fn test_validate_prefix_empty() {
        // 空文字
        assert!(validate_prefix("").is_err());
    }

    #[test]
    fn test_seckey_to_nsec() {
        // テスト用の秘密鍵（hex）
        let sk_hex = "3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d";
        let sk = SecretKey::from_slice(&hex::decode(sk_hex).unwrap()).unwrap();
        let nsec = seckey_to_nsec(&sk);

        // 正しい nsec（実装から生成された値）
        assert_eq!(nsec, "nsec180cvv07tjdrrgpa0j7j7tmnyl2yr6yr7l8j4s3evf6u64th6gkwsgyumg0");

        // nsec の形式が正しいことを確認
        assert!(nsec.starts_with("nsec1"));
        assert_eq!(nsec.len(), 63); // nsec1 + 58文字
    }

    #[test]
    fn test_pubkey_to_npub() {
        // テスト用の秘密鍵から公開鍵を生成
        let sk_hex = "3bf0c63fcb93463407af97a5e5ee64fa883d107ef9e558472c4eb9aaaefa459d";
        let sk = SecretKey::from_slice(&hex::decode(sk_hex).unwrap()).unwrap();
        let secp = Secp256k1::new();
        let pk = sk.public_key(&secp);

        let npub = pubkey_to_npub(&pk);

        // 正しい npub（実装から生成された値）
        assert_eq!(npub, "npub1wxxh2mmqeaghnme4kwwudkel7k8sfsrnf7qld4zppu9sglwljq5shd0y24");

        // npub の形式が正しいことを確認
        assert!(npub.starts_with("npub1"));
        assert_eq!(npub.len(), 63); // npub1 + 58文字
    }

    #[test]
    fn test_validate_prefix_error_messages() {
        // エラーメッセージの内容を確認
        let err = validate_prefix("abc").unwrap_err();
        assert!(err.contains("bech32 does not allow 'b'"));
        assert!(err.contains("excluded to avoid confusion"));

        let err = validate_prefix("test1").unwrap_err();
        assert!(err.contains("bech32 does not allow '1'"));
        assert!(err.contains("reserved as separator"));

        let err = validate_prefix("Test").unwrap_err();
        assert!(err.contains("uppercase letters"));
        assert!(err.contains("Use lowercase instead"));

        let err = validate_prefix("").unwrap_err();
        assert!(err.contains("cannot be empty"));
    }
}
