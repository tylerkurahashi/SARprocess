import numpy as np
from sarpy.io.phase_history.cphd import CPHDReader
import matplotlib.pyplot as plt
from scipy import signal
import time

def range_compression(cphd_file_path, channel_id=0):
    """
    最もシンプルなレンジ圧縮の実装

    Parameters:
    -----------
    cphd_file_path : str
        CPHDファイルのパス
    channel_id : int
        処理するチャネルのID（デフォルト: 0）

    Returns:
    --------
    range_compressed_data : ndarray
        レンジ圧縮されたデータ
    """
    # CPHD読み込み
    reader = CPHDReader(cphd_file_path)
    print("start range compression")

    # メタデータから必要な情報を取得
    # 1. TxWFParametersから送信波形の情報を取得（パルス長、チャープレート、中心周波数）
    tx_params = reader._cphd_meta.TxRcv.TxWFParameters[0]  # 最初の送信波形
    pulse_length = tx_params.PulseLength  # パルス長 [秒] - マッチトフィルタの長さを決定
    chirp_rate = tx_params.LFMRate  # チャープレート [Hz/s] - 周波数変化率
    freq_center = tx_params.FreqCenter  # 中心周波数 [Hz] - 搬送波の周波数

    # 2. RcvParametersから受信パラメータを取得（サンプリングレート）
    rcv_params = reader.cphd_meta.TxRcv.RcvParameters[0]  # 最初の受信パラメータセット
    sample_rate = rcv_params.SampleRate  # サンプリングレート [Hz] - 時間刻みの決定

    # 3. Channelsからデータサイズを取得
    num_pulses = reader.cphd_meta.Data.Channels[channel_id].NumVectors  # パルス数
    num_samples = reader.cphd_meta.Data.Channels[channel_id].NumSamples  # サンプル数

    # 生データの読み込み
    raw_data = reader.read()

    # レンジ参照信号（マッチトフィルタ）の生成
    # サンプル間隔（秒）
    dt = 1.0 / sample_rate

    # 参照信号の時間配列 - パルス長に応じたサンプル数
    num_ref_samples = int(pulse_length * sample_rate)
    t = np.arange(num_ref_samples) * dt

    # チャープ信号の複素共役（マッチトフィルタ）
    # チャープレートと中心周波数を使用して位相を計算
    phase = np.pi * chirp_rate * t**2 + 2 * np.pi * freq_center * t
    ref_signal = np.exp(1j * phase)

    # ウィンドウ関数の適用（サイドローブ低減のため）
    window = np.hamming(num_ref_samples)
    ref_signal = ref_signal * window

    # マッチトフィルタはチャープの複素共役
    matched_filter = np.conj(ref_signal)

    # FFTを使用した高速相関でレンジ圧縮を実行
    # FFTサイズの決定（2のべき乗に近い値）
    fft_size = 2 ** int(np.ceil(np.log2(num_samples + num_ref_samples - 1)))

    # 参照信号のFFT
    ref_fft = np.fft.fft(matched_filter, fft_size)

    # 各パルスを処理
    range_compressed_data = np.zeros((num_pulses, num_samples), dtype=complex)

    for i in range(num_pulses):
        # 現在のパルスデータ
        pulse_data = raw_data[i, :]

        # FFTで周波数領域に変換
        pulse_fft = np.fft.fft(pulse_data, fft_size)

        # 周波数領域で乗算（時間領域での畳み込みに相当）
        result_fft = pulse_fft * ref_fft

        # IFFTで時間領域に戻す
        result = np.fft.ifft(result_fft)

        # 必要な部分だけを保存
        range_compressed_data[i, :] = result[:num_samples]

    return range_compressed_data

if __name__ == "__main__":
    cphd_file_path = "../data/cphd/SkyFi_2421P7ON-2_2024-01-06_0025Z_SAR_VERY-HIGH_Ishikawa-Japan_CPHD.cphd"
    cphd = CPHDReader(cphd_file_path)
    data = cphd.read()
    print(data.shape)
    print(data[0][0])