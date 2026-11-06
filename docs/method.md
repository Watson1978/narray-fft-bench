# 比較の条件と手順

第一段階で決めたことだけを書く。題材ごとの条件は、各題材の段階で足す。

## 初回と定常

- 定常（`--mode steady`）: ウォームアップ 1 区間のあとの best-of-3。プランはすべてキャッシュに乗っている。`--inner` 回のステップを 1 区間とし、区間の終わりに 1 回だけ同期する。
- 初回（`--mode first`）: プロセスの中で、ステップを最初に 1 回呼んだときの時間。プランの生成と作業領域の確保を含む。1 プロセスで 1 点しか取れないので、`bench/run.sh` の 10 ラウンドの中央値で読む。
- 初回の前に `FFTBench.prime`（Python は `backend.prime`）で、長さ 3 の複素変換を 1 回回す。cuFFT の読み込み（cumo は最初の呼び出しで `dlopen` する。2.5〜4.3 ms）と CUDA の初期化を初回から外し、初回をその条件の形のプラン生成に絞るため。長さ 3 はどの条件も使わないので、キャッシュの 16 枚のうち 1 枚を埋めるだけになる。
- `bench_step` が返る前に走った変換（F の `fft(h, n: N)` のような固定のスペクトル）のプランは、初回に含まれない。
- Cumo の初回には、要素ごとの演算のカーネルを読み込む手間が入る。cumo は CUDA のモジュールを遅延して読み込み（`CUDA_MODULE_LOADING=LAZY`）、カーネルを型ごとに演算の族に分けて持つので、初めて使う族ごとに 3〜30 ms かかる。`prime` はその族の多くを読まない。D の初回では、Cumo の 37〜44 ms のうち約 30 ms がこれだった（[results/bignum_mul.md](results/bignum_mul.md) の「初回の内訳」）。`prime` は変えず、初回の表は、新しいプロセスで最初に払う時間として読む。
- cuFFT のプラン生成は数 ms〜数十 ms で、小さい変換の定常 1 回（数 µs）の 1000 倍になる。初回と定常を同じ表に混ぜない。`bench/aggregate.py` は、モードごとに別の表を出す。

## 同期

- 区間の終わりの同期は、Cumo が `cudaDeviceSynchronize`、CuPy が `Stream.null.synchronize()`。計画の §2 は「Cumo は結果のスカラー 1 個を読み戻す」としていたが、読み戻しは小さなカーネルと D2H のコピーを区間に足すので、narray-linalg-bench と同じく `cudaDeviceSynchronize` にした。どちらもデバイスの待ちを全部含む。
- `Cumo::FFT` は devInfo を読まないので、区間内に同期は無いはず。各題材の代表条件で nsys を使って数え、ここに書く（C2R の退避コピー、swapaxes のコピー、正規化のカーネルの本数も一緒に）。
- 数え方: `--rounds 1 --warmup 0` で `--inner` を 2 通りにした 2 つのプロファイル（`nsys profile -t cuda`）を取り、`nsys stats --force-export=true` の回数の差を、ステップ数の差で割る。準備と最後の同期は両方に同じだけ入るので消える。
- D（2026-10-18、fp32 の `fftconvolve` の n = 10⁵ と 10⁶、`fft_pow2` の n = 10⁶）: 区間内の同期（`cudaDeviceSynchronize`、`cudaStreamSynchronize`）は、Cumo も CuPy も 0 回。cuFFT のカーネル（R2C と C2R の前後の処理を含む）の種類と本数は、両者で同じ。`fftconvolve` の要素ごとのカーネルは、Cumo が 8 本（埋め草の `fill` 2、コピー 2、`store` 1、積、割り算、`rint`）、CuPy が 4 本と `cudaMemsetAsync` 2 回、`cudaMemcpyAsync` 2 回。`fft_pow2` では、Cumo の `real` がコピーのカーネルを 1 本足す。Cumo は変換ごとに `cudaEventRecord` を 1 回呼ぶ（プランを使い終えた印）。Cumo のカーネルの起動 1 回ごとに `cuKernelGetName` が 1 回出るが、cumo のソースには無い呼び出しで、CUDA ランタイムの中から出ている。
- E（2026-10-18、fp32 の n = 1024、`fft2` と `rfft2`、`--inner` 1000 と 3000 の差）: 区間内の同期は、Cumo も CuPy も 0 回。cuFFT のカーネルの種類と本数は同じ。`fft2` の cuFFT 以外のカーネルは、Cumo が 15 本、CuPy が 14 本。Cumo には、複素数を実数で割る前の型の変換と `real` のコピーの 2 本が多く、CuPy には、最大の位置を int64 にするコピーが 1 本多い。型の変換は、cumo 0.14.1 で無くなった（2026-10-22、[results/cumo-0.14.1.md](results/cumo-0.14.1.md)）。内訳は [results/phase_correlation.md](results/phase_correlation.md) の「Cumo が遅い理由」。`rfft2` では、Cumo の `irfft2` が入力を退避する `cudaMemcpyAsync`（デバイス内のコピー）を 1 回呼ぶ。
- G（2026-10-18、fp32）: Welch（L = 256、2 の冪、`--inner` 1000 と 3000 の差）は、区間内の同期が両者とも 0 回で、1 ステップのカーネルは Cumo 33 本（cuFFT 2 本）、CuPy 24 本（cuFFT 2 本）。Cumo の多い分は `z.real` と `z.imag` のコピー（4 本ずつ）。キャッシュを溢れさせる `welch_cycle`（`--inner` 20 と 60 の差）は、1 ステップに両者とも 20 枚のプランを作って捨てる（`cuModuleLoadData` と `cuModuleUnload` が 20 回ずつ）。Cumo はほかに、プランを捨てる前の `cudaEventSynchronize` を 20 回呼ぶ。
- F（2026-10-18、fp32 の `batch`、64 × 10000、`--inner` 1000 と 3000 の差）: 区間内の同期は、両者とも 0 回。1 ステップのカーネルは、Cumo が 9 本（cuFFT 2 本、割り算 3 本、コピー 3 本、`fill` 1 本）、CuPy が 9 本と `cudaMemsetAsync` 1 回。カーネルの時間の合計は Cumo 52 µs、CuPy 44 µs で、Cumo の 1 ステップ（99 µs）の差の大半は Ruby の minor GC だった（[results/ofdm.md](results/ofdm.md)）。
- B（2026-10-18）: nsys では数えていない。fp32 の N ≥ 1024 で Cumo が遅いのは、Ruby の GC がステップの時間のほとんどを占めるためだった（`GC.stat`、[results/taylor_green.md](results/taylor_green.md)）。
- C（2026-10-18、fp32 の `axes01`、512³、`--inner` 3 と 6 の差）: 区間内の同期は、両者とも 0 回。Cumo は末尾でない軸を転置のコピー（4 本）で末尾に移してから連続した配列を変換し、CuPy はストライドのまま cuFFT の advanced layout で変換する。cuFFT のカーネルの種類が違い（Cumo は `vector_fft_r2c` / `vector_fft_c2r`、CuPy は `regular_fft_r2c` / `regular_fft_c2r`）、CuPy の R2C と C2R は 42 ms と 74 ms かかる（[results/poisson3d.md](results/poisson3d.md)）。
- メモリの確保: `--inner` 10 と 30 の差では、n = 10⁵ の Cumo に `cudaMallocManaged` が 1 ステップに 1 回出た。1000 と 3000 の差では 0 回だった。立ち上がりの間だけ出るもので（Ruby の GC が一時配列を回収してプールに戻すまで、と見ている）、定常（ウォームアップ 1 区間のあと）には入らない。CuPy は 10 と 30 の差でも 0 回。

## プランキャッシュ

- CuPy は既定の大きさ（デバイスごとに 16）のまま。`python/bench.py` の 2 列目に、キャッシュの今の枚数と大きさを出す。
- cumo はスレッドごと・GPU ごとに 16。`script/bench.rb` の 2 列目に、`Cumo::CUDA::Cufft.plan_cache_info` のキャッシュの今の枚数、大きさ、プロセスの中での失敗（プランを作った）回数を出す（2026-10-18 から）。
- cuFFT の作業領域の持ち方が違う。CuPy はプランごとに持ち続け、cumo は変換を呼ぶたびにメモリプールから借りて返す（[cupy-notes.md](cupy-notes.md)）。定常では、cumo の区間に、プールから借りて返す手間が呼び出しごとに入る。計画の §5 は「同じ土俵」としていたが、そうではない。
- 計測は 1 スレッド・1 GPU で行う。

## CPU 表

- Numo は numo-pocketfft 0.7.0（`FFTBench::NumoFFT` 経由）、Python は numpy 2.5.3 と scipy 1.18.1。どちらも pocketfft だが、numo-pocketfft は C 版で、呼び出しごとにプランを作って捨てる（`ext/numo/pocketfft/pocketfftext.c` で確認）。numpy は C++ 版で、プランをキャッシュするかは未確認。配布物の `_pocketfft_umath` は記号が落とされていて読めず、2026-10-17 に時間で見たところ、長さ 65537 では初回だけ遅く（9.7 ms → 5.8 ms）、75600 では差が無かった。CPU 表の初回を読む前に、numpy のソースで確かめる。
- numo-pocketfft は倍精度しか無いので、CPU 表は fp64 だけにする。
