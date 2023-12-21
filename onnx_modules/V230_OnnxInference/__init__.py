import numpy as np
import onnxruntime as ort


def convert_pad_shape(pad_shape):
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = np.arange(max_length, dtype=length.dtype)
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, -1)

    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y)
    path = path.reshape(b, t_x, t_y)
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    return path


class OnnxInferenceSession:
    def __init__(self, path, Providers=["CPUExecutionProvider"]):
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)
        self.sdp = ort.InferenceSession(path["sdp"], providers=Providers)
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)

    def __call__(
        self,
        seq,
        tone,
        language,
        bert_zh,
        bert_jp,
        bert_en,
        sid,
        seed=114514,
        seq_noise_scale=0.8,
        sdp_noise_scale=0.6,
        length_scale=1.0,
        sdp_ratio=0.0,
    ):
        if seq.ndim == 1:
            seq = np.expand_dims(seq, 0)
        if tone.ndim == 1:
            tone = np.expand_dims(tone, 0)
        if language.ndim == 1:
            language = np.expand_dims(language, 0)
        assert (seq.ndim == 2, tone.ndim == 2, language.ndim == 2)
        g = self.emb_g.run(
            None,
            {
                "sid": sid.astype(np.int64),
            },
        )[0]
        g = np.expand_dims(g, -1)
        enc_rtn = self.enc.run(
            None,
            {
                "x": seq.astype(np.int64),
                "t": tone.astype(np.int64),
                "language": language.astype(np.int64),
                "bert_0": bert_zh.astype(np.float32),
                "bert_1": bert_jp.astype(np.float32),
                "bert_2": bert_en.astype(np.float32),
                "g": g.astype(np.float32),
            },
        )
        x, m_p, logs_p, x_mask = enc_rtn[0], enc_rtn[1], enc_rtn[2], enc_rtn[3]
        np.random.seed(seed)
        zinput = np.random.randn(x.shape[0], 2, x.shape[2]) * sdp_noise_scale
        logw = self.sdp.run(
            None, {"x": x, "x_mask": x_mask, "zin": zinput.astype(np.float32), "g": g}
        )[0] * (sdp_ratio) + self.dp.run(None, {"x": x, "x_mask": x_mask, "g": g})[
            0
        ] * (
            1 - sdp_ratio
        )
        w = np.exp(logw) * x_mask * length_scale
        w_ceil = np.ceil(w)
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)), a_min=1.0, a_max=100000).astype(
            np.int64
        )
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = (
            m_p
            + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])
            * np.exp(logs_p)
            * seq_noise_scale
        )

        z = self.flow.run(
            None,
            {
                "z_p": z_p.astype(np.float32),
                "y_mask": y_mask.astype(np.float32),
                "g": g,
            },
        )[0]

        return self.dec.run(None, {"z_in": z.astype(np.float32), "g": g})[0]
