import argparse
from multiprocessing import cpu_count

from gradio_tabs.train import preprocess_all
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict


# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", "-m", type=str, help="Model name", required=True
    )
    parser.add_argument("--batch_size", "-b", type=int, help="Batch size", default=2)
    parser.add_argument("--epochs", "-e", type=int, help="Epochs", default=100)
    parser.add_argument(
        "--save_every_steps",
        "-s",
        type=int,
        help="Save every steps",
        default=1000,
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="Number of processes",
        default=cpu_count() // 2,
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Loudness normalize audio",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Trim silence",
    )
    parser.add_argument(
        "--freeze_EN_bert",
        action="store_true",
        help="Freeze English BERT",
    )
    parser.add_argument(
        "--freeze_JP_bert",
        action="store_true",
        help="Freeze Japanese BERT",
    )
    parser.add_argument(
        "--freeze_ZH_bert",
        action="store_true",
        help="Freeze Chinese BERT",
    )
    parser.add_argument(
        "--freeze_style",
        action="store_true",
        help="Freeze style vector",
    )
    parser.add_argument(
        "--freeze_decoder",
        action="store_true",
        help="Freeze decoder",
    )
    parser.add_argument(
        "--use_jp_extra",
        action="store_true",
        help="Use JP-Extra model",
    )
    parser.add_argument(
        "--val_per_lang",
        type=int,
        help="Validation per language",
        default=0,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="Log interval",
        default=200,
    )
    parser.add_argument(
        "--yomi_error",
        type=str,
        help="Yomi error. Options: raise, skip, use",
        default="raise",
    )

    args = parser.parse_args()

    preprocess_all(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_every_steps=args.save_every_steps,
        num_processes=args.num_processes,
        normalize=args.normalize,
        trim=args.trim,
        freeze_EN_bert=args.freeze_EN_bert,
        freeze_JP_bert=args.freeze_JP_bert,
        freeze_ZH_bert=args.freeze_ZH_bert,
        freeze_style=args.freeze_style,
        freeze_decoder=args.freeze_decoder,
        use_jp_extra=args.use_jp_extra,
        val_per_lang=args.val_per_lang,
        log_interval=args.log_interval,
        yomi_error=args.yomi_error,
    )
