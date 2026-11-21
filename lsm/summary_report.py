"""
Genera tablas y gráficas comparativas consolidadas entre todos los enfoques entrenados
(Estáticos: MLP-Base, HandGeoResNet, MobileNetV3-Small; Dinámicos: LSTM-Base, DynamicBiGRUAttn)
para incluir directamente en el mini paper.

Uso:
  python lsm/summary_report.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lsm.train_static_cnn import plot_cnn_results


def _annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.35, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=7.8)


def main():
    out = Path("outputs")
    st = json.loads(Path("outputs/static/resultados.json").read_text(encoding="utf-8"))
    cnn = json.loads(Path("outputs/static_cnn/resultados.json").read_text(encoding="utf-8"))
    dy = json.loads(Path("outputs/dynamic/resultados.json").read_text(encoding="utf-8"))

    mlp, geo = st["mlp_base"], st["georesnet"]
    lstm, gru = dy["lstm_base"], dy["bigru_attn"]

    # 1. Tabla consolidada CSV (incluye exactitud condicional sobre detecciones y efectiva sobre 100%)
    rows = [
        [
            "Estático (21 letras)", "MLP-Base (60D crudas)", mlp["params"],
            f"{mlp['test_accuracy']:.4f}", f"{mlp['mediapipe_detection_rate']:.4f}",
            f"{mlp['test_effective_accuracy']:.4f}", f"{mlp['test_f1_macro']:.4f}", f"{mlp['val_f1_macro']:.4f}",
            f"{mlp['by_subset']['A']['acc_detected']:.4f}", f"{mlp['by_subset']['B']['acc_detected']:.4f}", f"{mlp['by_subset']['C']['acc_detected']:.4f}",
            f"{mlp['by_subset']['A']['acc_effective']:.4f}", f"{mlp['by_subset']['B']['acc_effective']:.4f}", f"{mlp['by_subset']['C']['acc_effective']:.4f}",
            f"{mlp['training_seconds']:.1f}",
        ],
        [
            "Estático (21 letras)", "MobileNetV3-Small (CNN RGB)", cnn["params"],
            f"{cnn['test_accuracy']:.4f}", "1.0000",
            f"{cnn['test_accuracy']:.4f}", f"{cnn['test_f1_macro']:.4f}", f"{cnn['val_f1_macro']:.4f}",
            f"{cnn['by_subset']['A']['acc']:.4f}", f"{cnn['by_subset']['B']['acc']:.4f}", f"{cnn['by_subset']['C']['acc']:.4f}",
            f"{cnn['by_subset']['A']['acc']:.4f}", f"{cnn['by_subset']['B']['acc']:.4f}", f"{cnn['by_subset']['C']['acc']:.4f}",
            f"{cnn['training_seconds']:.1f}",
        ],
        [
            "Estático (21 letras)", "HandGeoResNet (184D geom.)", geo["params"],
            f"{geo['test_accuracy']:.4f}", f"{geo['mediapipe_detection_rate']:.4f}",
            f"{geo['test_effective_accuracy']:.4f}", f"{geo['test_f1_macro']:.4f}", f"{geo['val_f1_macro']:.4f}",
            f"{geo['by_subset']['A']['acc_detected']:.4f}", f"{geo['by_subset']['B']['acc_detected']:.4f}", f"{geo['by_subset']['C']['acc_detected']:.4f}",
            f"{geo['by_subset']['A']['acc_effective']:.4f}", f"{geo['by_subset']['B']['acc_effective']:.4f}", f"{geo['by_subset']['C']['acc_effective']:.4f}",
            f"{geo['training_seconds']:.1f}",
        ],
        [
            "Dinámico (6 letras)", "LSTM-Base (63D secuencia)", lstm["params"],
            f"{lstm['test_accuracy']:.4f}", "1.0000",
            f"{lstm['test_accuracy']:.4f}", f"{lstm['test_f1_macro']:.4f}", f"{lstm['val_f1_macro']:.4f}",
            "-", "-", "-", "-", "-", "-", f"{lstm['training_seconds']:.1f}",
        ],
        [
            "Dinámico (6 letras)", "DynamicBiGRUAttn (214D esp-temp)", gru["params"],
            f"{gru['test_accuracy']:.4f}", "1.0000",
            f"{gru['test_accuracy']:.4f}", f"{gru['test_f1_macro']:.4f}", f"{gru['val_f1_macro']:.4f}",
            "-", "-", "-", "-", "-", "-", f"{gru['training_seconds']:.1f}",
        ],
    ]
    with open(out / "tabla_comparativa_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "tarea", "modelo", "parametros", "test_acc_detectado", "deteccion_mp",
            "test_acc_efectivo", "test_f1_macro", "val_f1_macro",
            "acc_A_det", "acc_B_det", "acc_C_det",
            "acc_A_efec", "acc_B_efec", "acc_C_efec", "seg_entrenamiento",
        ])
        w.writerows(rows)

    # 2. Gráfica estática comparativa de 2 paneles: condicional vs efectiva (100% imágenes)
    labels = ["Global", "Sub A\n(baja var.)", "Sub B\n(rotación)", "Sub C\n(extrema)"]
    x = np.arange(len(labels))
    w_bar = 0.27

    panels = [
        (
            "Exactitud sobre detecciones (MediaPipe 98.95% / Sub C 97.27%)\nvs. CNN RGB (100%)",
            [mlp["test_accuracy"], mlp["by_subset"]["A"]["acc_detected"], mlp["by_subset"]["B"]["acc_detected"], mlp["by_subset"]["C"]["acc_detected"]],
            [cnn["test_accuracy"], cnn["by_subset"]["A"]["acc"], cnn["by_subset"]["B"]["acc"], cnn["by_subset"]["C"]["acc"]],
            [geo["test_accuracy"], geo["by_subset"]["A"]["acc_detected"], geo["by_subset"]["B"]["acc_detected"], geo["by_subset"]["C"]["acc_detected"]],
        ),
        (
            "Exactitud efectiva sobre el 100% de imágenes (27,619)\n(no-detecciones cuentan como error)",
            [mlp["test_effective_accuracy"], mlp["by_subset"]["A"]["acc_effective"], mlp["by_subset"]["B"]["acc_effective"], mlp["by_subset"]["C"]["acc_effective"]],
            [cnn["test_accuracy"], cnn["by_subset"]["A"]["acc"], cnn["by_subset"]["B"]["acc"], cnn["by_subset"]["C"]["acc"]],
            [geo["test_effective_accuracy"], geo["by_subset"]["A"]["acc_effective"], geo["by_subset"]["B"]["acc_effective"], geo["by_subset"]["C"]["acc_effective"]],
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    for ax, (title, v_mlp, v_cnn, v_geo) in zip(axes, panels):
        b1 = ax.bar(x - w_bar, [v * 100 for v in v_mlp], w_bar, label="MLP-Base (60D)", color="#8da0cb")
        b2 = ax.bar(x,         [v * 100 for v in v_cnn], w_bar, label="MobileNetV3-Small (CNN RGB)", color="#fc8d62")
        b3 = ax.bar(x + w_bar, [v * 100 for v in v_geo], w_bar, label="HandGeoResNet (184D geom.)", color="#1b9e77")
        _annotate_bars(ax, b1)
        _annotate_bars(ax, b2)
        _annotate_bars(ax, b3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(80, 103.5)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Exactitud en prueba (personas S18 y S20 no vistas) [%]")
    axes[1].legend(loc="upper right", fontsize=8.8)
    fig.suptitle("Comparación justa de enfoques en LSM Estático por nivel de variación angular", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(out / "comparacion_global.png", dpi=160)
    plt.close(fig)

    # 3. F1 por letra: HandGeoResNet vs MobileNetV3-Small
    classes = st["classes"]
    x2 = np.arange(len(classes))
    w2 = 0.38
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(x2 - w2 / 2, [cnn["per_class_f1"][c] for c in classes], w2,
           label=f"MobileNetV3-Small CNN (F1={cnn['test_f1_macro']:.3f})", color="#fc8d62")
    ax.bar(x2 + w2 / 2, [geo["per_class_f1"][c] for c in classes], w2,
           label=f"HandGeoResNet Landmarks (F1={geo['test_f1_macro']:.3f})", color="#1b9e77")
    ax.set_xticks(x2)
    ax.set_xticklabels(classes)
    ax.set_ylim(0.80, 1.02)
    ax.set_ylabel("F1-Score en prueba")
    ax.set_title("F1-Score por letra: CNN RGB vs Landmarks Geométricos (21 señas estáticas)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "comparacion_f1_landmarks_vs_cnn.png", dpi=160)
    plt.close(fig)

    # 4. Gráficas faltantes de MobileNetV3-Small a partir de resultados.json y matriz_confusion_cnn.csv
    cm_cnn = np.loadtxt(out / "static_cnn" / "matriz_confusion_cnn.csv", delimiter=",", dtype=int)
    plot_cnn_results(
        out / "static_cnn",
        classes,
        cnn["history"],
        int(cnn["best_epoch"]),
        cm_cnn,
        float(cnn["test_accuracy"]),
    )
    print("Resumen consolidado y gráficas CNN generados en outputs/")


if __name__ == "__main__":
    main()
