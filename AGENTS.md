# AGENTS.md

このリポジトリは `Cumo::FFT`（cuFFT 実装）を、**numo-pocketfft と互換か（計算結果を含む）** と **CuPy と比べて速いか** の 2 点で検証するプロジェクト。進め方は [docs/plans/narray-fft-bench-plan.md](docs/plans/narray-fft-bench-plan.md) に従う。**前の段階の受け入れテストが通るまで、次の段階のコードを書かないこと。** 作法の多くは narray-linalg-bench の AGENTS.md（さらに narray-llm の AGENTS.md）から引き継いでいる。ここには FFT で効くものを置く。

## 段階

| 段階 | 内容 | 状態 |
|---|---|---|
| 第一段階 | 足場（backend / Numo シム / fixtures / gates / timer、7 題材の small fixture、Ruby と Python のバイト一致） | 完了（2026-10-17） |
| 第二〜第八段階 | 題材を D → E → G → F → B → C → A の順に 1 つずつ | 7 題材とも完了（2026-10-18）。A の narray-llm への rfft の差し込みは、narray-llm 側の判断で見送った（DFT の表の直しだけを取り込んだ） |
| 第九段階 | `Cumo::Pocketfft` 互換層の試験 | 完了（2026-10-18、E より先に行った。[docs/results/pocketfft-compat.md](docs/results/pocketfft-compat.md)） |
| 第十段階 | 計測の本番 | 題材ごとに第二〜第八段階の中で行い、README にまとめた（2026-10-18）。E、B、C は、2026-10-22 に cumo 0.14.1 で測り直した。2026-11-03 に、cumo 0.15.0 で回帰が無いことを確かめた |
| 第十一段階 | 見つけたものを cumo に戻す | docs/cumo-issues.md にまとめ、cumo 側のセッションに送った（2026-10-18） |

2026-10-18: cumo 0.14.0 のリリースは、D と第九段階の結果を待つことになった（両方のユーザーの判断）。そのため、第九段階を D のすぐあと、E より前に行う。E〜A で見つかったものは、0.14.1 以降に回す。

## バックエンドの切り替え

既定は Numo（CPU）で、`GPU=1` を付けると Cumo になる。dtype は `DTYPE=fp32`（既定）か `DTYPE=fp64` で選ぶ。`script/gate.rb`、`python/gate.py`、テストは、`DTYPE` が無ければ fp64 で動く。

```ruby
XM  = Cumo or Numo                    # lib/fft_bench/backend.rb
XFT = Cumo::FFT or FFTBench::NumoFFT  # numpy.fft の関数と引数
XF  = XM::SFloat or XM::DFloat
XC  = XM::SComplex or XM::DComplex    # XF に対応する複素数の型
HM  = Numo                            # fixture の読み書きと生成だけが使う
```

- **題材コード（`lib/fft_bench/problems/`）には `Numo::` / `Cumo::` / `Pocketfft` を直接書かない。** 使うのは `XM`、`XFT`、`XF`、`XC` だけ。Python 側（`python/problems/`）も同じで、使うのは `xp` / `xfft` / `xsig`（`python/backend.py`）だけにする。`numpy.` / `cupy.` は書かない。
- 呼び方は `Cumo::FFT` に合わせる。`XFT.fft(a, n: nil, axis: -1, norm: nil)`、`XFT.fftn(a, s: nil, axes: nil, norm: nil)`、`XFT.fftfreq(n, d: 1.0)`（`d` はキーワード）、`XFT.fftshift(x, axes: nil)`、`XFT.fftconvolve(a, b)`（full のみ）。
- **`lib/fft_bench.rb` は `bundler/setup` を読む。** この環境には numo-narray 0.9.2.1 と numo-narray-alt 0.11.2 の両方が入っている。bundler を通さずに読むと 0.9.2.1 が読まれ、定数の再定義警告が大量に出る。
- CuPy は `CUPY_TF32=0` で動かす（`python/backend.py` が既定で設定する）。FFT には効かないが、D と F の `dot` には効く。

## Numo シム（`lib/fft_bench/numo_fft.rb`）

- numo-pocketfft 0.7.0 の公開関数は、`fft` 系が 1 次元だけ、`fft2` 系が 2 次元だけを受け、`n:` / `axis:` / `norm:` が無い。`irfft` の出力長は常に `2(m − 1)` で、奇数長を出せない。
- **シムは private の `ext_cfft` / `ext_icfft` / `ext_rfft` / `ext_irfft` を `send` で呼ぶ。** これらは任意の形の配列の末尾の軸を、行ごとに 1 回の呼び出しで変換する（numo-pocketfft 自身の `fftn` もこれで組まれている）。公開関数で書くとバッチが Ruby のループになり、CPU 表がループの速さを測ってしまう。読み込み時に 4 つがあるかを確かめ、無ければ `LoadError` で止まる。
- `ext_irfft` は、出力長 n の複素数の行の先頭 n/2 + 1 個を半スペクトルとして読み、n の偶奇によらず n 点を返す。`ext_icfft` と `ext_irfft` は 1/n を掛けて返し、`ext_cfft` と `ext_rfft` は掛けない。`norm:` はその差を Ruby で掛けて埋める。
- N 次元の関数は numpy の `_raw_fftnd` と同じく、軸ごとに 1 次元の変換を、最後の軸から順に掛ける。重複した軸と `s:` の `-1` も numpy と同じに扱う。
- pocketfft は倍精度しか無いので、シムの結果は常に DComplex / DFloat になる。CPU 表が fp64 だけなのはこのため。
- `fftconvolve` は `Numo::Pocketfft.fftconvolve` をそのまま呼ぶ（複素数の `fftn` で、2 の冪への切り上げも rfft も使わない）。scipy は実数なら rfft を使い、長さを速い長さに切り上げるので、CPU 表のこの比は実装の差として読む。
- **シムは numpy の参照と突き合わせてある。** `test/test_numo_fft.rb` が 1681 ケース（1 次元 6 関数の n / axis / norm の全組み合わせ、N 次元 8 関数の s / axes / norm、周波数、シフト、`fftconvolve`）を比べる。最大の相対誤差は 7.5e-16（`ifft2`）で、tol はその 10 倍の 1e-14。周波数とシフトはビット一致を求める。

## cumo の版

**`Cumo::FFT` と `Cumo::Pocketfft` は、cumo 0.14.0（2026-10-18 リリース、タグ `v0.14.0` = `afd8ead9`）で初めてリリースに入った。** 既定は、0.15.0（2026-11-02 リリース、タグ `v0.15.0` = `ee6facf2`）の gem を sm_120 向けにビルドした `vendor/cumo-0.15.0/`。どのビルドを読んだかは、`script/bench.rb` の 2 列目に出る `cumo.so` のパスで確かめる。`Cumo::REVISION` は `-I` で読むと nil になり、0.14.0 の gem では `"v0.14.0"` ではなく `"v0.13.0-11-gafd8ead9"` を返す（タグより先に gem を作ったため。cumo 側のセッションからの連絡）ので、見分けに使わない。0.14.1 からの gem は版のタグ（`"v0.14.1"`、`"v0.15.0"`）を返す（#626）が、`-I` で読むと nil になるのは同じ。

- **cumo は Gemfile に入れず、共有の gem 置き場にも入れない。** 共有の置き場は narray-llm と narray-linalg-bench も使う。2026-10-01 には、master を共有の置き場に入れて、同じ版の文字列のリリース版を上書きし、両方の環境を壊した（narray-linalg-bench の AGENTS.md）。0.14.0 は、2026-10-18 にユーザーが共有の置き場（`~/.rbenv/versions/4.0.7/lib/ruby/gems/4.0.0`）に入れた。0.9.0〜0.13.0 も並んで残っている。このリポジトリは、引き続き `vendor/` のビルドを読む。
- ビルドは `vendor/`（git には入れない）に置く。`GPU=1` のときは `FFTBench.add_cumo_load_path` がその lib を `$LOAD_PATH` の先頭に足す。別のビルドと比べるときは `CUMO_LIB=<lib のパス>` で差し替える。既定を変えるときは `lib/fft_bench/backend.rb` の `DEFAULT_CUMO` を書き換え、この節に記録する。
- リリースの gem の入れ方: `CUMO_NVCC_GENERATE_CODE=arch=compute_120,code=sm_120 gem install cumo -v <版> --no-document --ignore-dependencies --install-dir vendor/cumo-<版>`（2026-10-18 の 0.14.0 で 3 分）。
- master のビルドの入れ方: cumo を scratchpad に `git clone --no-hardlinks` し、目的のコミットで `rake build` を実行する（作業ツリーに別の作業が残っていることがあるため）。そのあと `CUMO_NVCC_GENERATE_CODE=arch=compute_120,code=sm_120 gem install --local --no-document --ignore-dependencies --install-dir vendor/cumo-<コミット> pkg/cumo-<版>.gem` を実行する。master の版の文字列は直前のリリースと同じなので、`Cumo::VERSION` では区別できない。
- 2026-10-17: master `a3910f07`（#620〜#622）を `vendor/cumo-a3910f07/` に入れた。`Cumo::Pocketfft` と `fftconvolve` はまだ無い。
- 2026-10-17: master `e9ae5d76`（#623 まで）を `vendor/cumo-e9ae5d76/` に入れ、既定にした。D、E、G の計測と、第九段階の互換の試験は、このビルドで行った。
- 2026-10-18: 0.14.0 を `vendor/cumo-0.14.0/` に入れ、既定をこれに替えた。中身は master `ca4151f9` と同じで、`e9ae5d76` のあとに入ったのは、テストだけの修正（#624）と、ユーザーのカーネルの修正（#625。FFT の経路は触っていない）。D、E、G、F の Cumo の関門の値と `values` は、fp64 と fp32 の両方で、`e9ae5d76` とビットまで同じだった。テスト全体も通った。
- 2026-10-22: 0.14.1 を `vendor/cumo-0.14.1/` に入れ、既定をこれに替えた（ビルドは 3 分）。題材に効く変更は #632 で、同じ精度の複素数と実数の + − × ÷ が、実数を複素数に変換しなくなった。7 題材の関門の値と `values` は、fp64 と fp32 の両方で 0.14.0 とビットまで同じだった。`GPU=1 rake test` も通った。E、B、C が速くなったので、この 3 題材の GPU の表を測り直した。A、D、F、G の表は前のビルドのまま（[docs/results/cumo-0.14.1.md](docs/results/cumo-0.14.1.md)）。
- 2026-11-03: 0.15.0 を `vendor/cumo-0.15.0/` に入れ、既定をこれに替えた（ビルドは 3 分）。FFT のコードは変わっていない。7 題材の関門の値と `values` は、fp64 と fp32 の両方で 0.14.1 とビットまで同じだった。`rake test` も CPU と `GPU=1` で通った。0.14.1 と同じラウンドで交互に測ると、定常の 25 条件は変わらなかった。初回は、ステップが縮約を使う A、E、G が 1.15〜1.60 倍速くなった。縮約のカーネルを分けた #634 の効果と見ている。題材ごとの表は前のビルドのまま（[docs/results/cumo-0.15.0.md](docs/results/cumo-0.15.0.md)）。
- **GPU で計測を始める前に、cumo 側のセッションに一声かける。** cumo 側も、重い作業（全体のビルド、メモリを使い切る実験）の前にこちらへ声をかける。

## fixture

- **乱数を引くのは `script/make_fixtures.rb` の 1 か所だけ。** Numo の DFloat で、題材ごとに固定した seed で作る。Numo、Cumo、numpy の乱数は同じ seed でも一致しないので、4 実装とも `data/` の同じバイトを読む。スペクトルの整形（B の乱数初期場、E のテクスチャと小数シフト）も、ホストの `FFTBench::NumoFFT` でここで行う。
- 形式は `data/<題材>_<サイズ>.bin`（リトルエンディアンの fp64、行優先、配列を順に連結）と `data/<題材>_<サイズ>.json`（各配列の offset、shape、sha256、スカラーの属性）の 2 ファイル。**複素数の配列は実部と虚部の 2 つのブロックに分けて書く**（`"part": "re"` / `"im"`）。読み側で組み立てるので、複素数型のバイト配置に依存しない。
- `data/shim_small` はシムの試験の入力とケースの一覧、`data/shim_ref_small` はその numpy / scipy の参照（`python/make_shim_refs.py` が作る。numpy の非推奨の警告はエラーにしてある）。
- A のメルフィルタ（80 × 201）は、openai/whisper-tiny の `preprocessor_config.json` から取り出した `script/whisper_mel_filters.json` を追跡している。A の波形は narray-llm の Whisper の試験と同じ式（チャープ + 440 Hz、包絡付き、fp32 に丸める）を Numo で計算する。
- `data/` は git に入れない。`rake fixtures` で作り直す。

## 関門（正しさ）

- **fp64 で回す。** 相対誤差は max ノルムで `‖x − x_ref‖_max / ‖x_ref‖_max`。tol は fp64 で 1e-12、fp32 で 1e-5 から始める。**Numo で実際の誤差を測り、`docs/results/<題材>.md` に書いてから** その 10 倍に固定する。
- **完全一致が取れる題材（A・D・E・F）では tol を持ち込まない。** `Gates.mismatches` で、違う要素の数が 0 であることを求める。
- 丸めの境目（D の係数の誤差が 0.5 未満）のような「未満」の関門は `Gates.lt` で書く。fp32 では、誤差がちょうど 0.5 になり、偶数丸めで外れることがある。
- 補助の関門として、Parseval（`Gates.parseval`。`norm:` の 3 種で係数が 1/n、1、n に変わる。片側スペクトルは `half:` で DC と Nyquist 以外を 2 倍に数える）と往復の恒等を使う。
- 「Numo と一致した」ではなく、**「正しい答えを出した」で判定する。** Numo と Cumo の直接比較は、関門とは別に `test/test_parity.rb` と `test/test_pocketfft_parity.rb` で行う（cumo が読めない環境では omit）。`test/test_parity.rb` は、題材が呼ぶ形の呼び出しを fp64 で比べる。tol は測った差の最大値の 10 倍で、D の呼び出しでは 1.42e-15（長さの違う 2 つの `fftconvolve`）から 1.5e-14 にし、F で 1.61e-15（2 次元のバッチの `fftconvolve`）が出たので 1.6e-14 にした。題材を足すたびに、その題材の呼び出しを足して測り直す。`test/test_pocketfft_parity.rb` の tol は、倍精度が 5e-15、単精度が 7.5e-7（測った差の最大 4.93e-16 と 7.33e-8 の 10 倍）。**0 次元の配列で numo-pocketfft の `fftn` / `rfftn` を呼ばない。** C の拡張が配列の外を読み、プロセスの状態で結果が変わり、返らなくなることもある。4 実装の物理量（`values`）が揃っているかは、`script/compare_gates.rb` で並べて確かめる。

## Numo と Cumo の差異

- **Cumo の縮約は 0 次元 NArray を返す。** 読み戻しは `FFTBench.scalar` の 1 か所でだけ行う。**反復ループ（B・C の時間発展、F のシンボル列）の中では読み戻さない。**
- `0d_narray > x` は Bit を返し、偽でも真として扱われる。テストで比べるときは、`scalar` で Ruby の数値にしてから比べる。
- Bit 配列、マスク代入、NArray を使った添字は、題材コードでは避ける。Cumo は NArray の添字を同期して読み戻す。
- `max_index` は平坦化した通し番号を返す（E の argmax）。
- NArray どうしの `==` は、要素ごとではなく真偽を 1 つだけ返す。テストでは `to_a` か `to_binary` どうしで比べる。
- `Numo::DFloat` には `real` / `imag` が無い。複素数かどうかは `FFTBench.complex?` で分けてから呼ぶ。

## 計測の作法

- `script/bench.rb` と `python/bench.py` は、1 題材 × 1 サイズ × 1 dtype × 1 モードを測り、TSV を 1 行出す。6 列目が ms/step、9 列目がモード。
- **初回（`--mode first`）と定常（`--mode steady`、既定）を分け、同じ表に混ぜない。** 定常は、ウォームアップ 1 回のあとの best-of-3。初回は、`FFTBench.prime`（長さ 3 の変換でライブラリの読み込みと CUDA の初期化を済ませる）のあとの、ステップの最初の 1 回。`bench_step` の準備で走った変換のプランは、初回に含まれない。詳しくは [docs/method.md](docs/method.md)。
- Cumo の一時配列は、Ruby の GC が回るまでメモリプールに戻らない。cumo は GPU のメモリの確保を `rb_gc_adjust_memory_usage` で Ruby に知らせるので、数 MB の配列を毎ステップ作ると、minor GC がほぼ毎ステップ走る（F の 64 × 10000 本で 1 回 60〜80 µs）。B の fp32 の N = 2048 では、片側スペクトルの配列 1 本（16.8 MB）が malloc の上限（16 MB）を超え、1 ステップの 12.8 ms のうち 12.0 ms が GC だった（0.14.0。0.14.1 では、混ぜた演算の一時配列が減って 11.8 ms のうち 11.0 ms）。CuPy との差を読むときは、`GC.stat` の回数と時間も見る（[docs/results/ofdm.md](docs/results/ofdm.md)、[docs/results/taylor_green.md](docs/results/taylor_green.md)）。計測の表は既定の GC の設定で取る。
- Cumo の初回には、要素ごとの演算のカーネルを読み込む手間（初めて使う型と演算の族ごとに 3〜30 ms）が入る。cumo は CUDA のモジュールを遅延して読み込むため。D では、初回の約 30 ms がこれだった。初回の表は、新しいプロセスで最初に払う時間として読み、プランの生成の差として読まない（[docs/method.md](docs/method.md)）。
- 区間内では同期せず、区間の終わりに 1 回だけ同期する（Cumo は `cudaDeviceSynchronize`、CuPy は `Stream.null.synchronize()`）。fixture の読み込みとプロセスの起動は区間に含めない。
- `bench/conditions.tsv` は 1 行が「名前、モード、Ruby のコマンド、Python のコマンド」。`bench/run.sh` がモードを `--mode` として両方のコマンドに足し、出力の 9 列目がそれと違えば止まる。
- **README に載せる数字は `bench/run.sh` の 10 ラウンド（対照系列付き）から取ったものだけにする。** 対照が 1 をまたがない条件は載せない。単発の数字は書かない。
- 1 プロセスが 2 秒以上回るように `--inner` を選び、`bench/conditions.tsv` に書く。
- GPU の表（`bench/conditions.tsv`）と CPU の表（`bench/conditions_cpu.tsv`）を混ぜない。
- **クロックの固定は再起動で外れる。計測の直前に、毎回読んで確かめる。** アイドル時の SM が 2790〜2805 MHz なら固定されている。GPU 条件では、`bench/run.sh` が読んで、外れていれば止まる。メモリクロックの段、帰属、読み方、nsys の注意は narray-linalg-bench の AGENTS.md「計測の作法」をそのまま守る。
- 検証や計測のスクリプトは単一プロセスで回すか、並列数を絞る（`xargs -P2`）。子プロセスにはすべて `timeout` を付ける。ベンチのスクリプトは bash で書く。ログインシェルは zsh で、zsh は変数に入れた引数列を単語に分割しない。
- `/tmp` は tmpfs で、再起動すると scratchpad ごと消える。残す数字は、その日のうちに `docs/results/` に書く。

## テストの作法

- **GPU が無くても `rake test` が全部通る状態を保つ。** テストは Numo と numpy で動く。`GPU=1` を付けると、`XM` / `XFT` を使うテストは Cumo で動く。
- Python の venv（`python/.venv`）が無いときは、Python 側のテストとシムの試験は omit になる。
- 再現用の一時スクリプトは scratchpad に置く。リグレッションテストとしてコミットするかどうかは、その都度確認する。

## 環境

```
bundle install
python3 -m venv python/.venv
python/.venv/bin/pip install -r python/requirements.txt
rake fixtures
rake test
```

GPU を使うときは、上の「cumo の版」の手順で `vendor/` に cumo master を入れる。
