import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import openpyxl
import numpy as np
from collections import Counter
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, roc_curve, recall_score, precision_score, f1_score


####

#Here I'll be working with a structure for each
#model that is
# {
#     name - string - name of the model
#     y_pred - list of predictions
#     scaling - either None or StandardScaler
#     cv - either None or the Pipeline object
# }

####


def append_first_row(ws):
    ws.append([
        "Model",
        "Scaling",
        "Num Features",
        "Best Params",
        "Precision (macro)",
        "Δ Precision (%)",
        "Recall (macro)",
        "Δ Recall (%)",
        "F1 (macro)",
        "Δ F1 (%)"
    ])

def base_model(y, ws):

    majority_class = Counter(y).most_common(1)[0][0]
    y_pred = np.full_like(y, majority_class)

    base_model_recall = recall_score(y, y_pred, average="macro", zero_division=0)
    base_model_precision = precision_score(y, y_pred, average="macro", zero_division=0)
    base_model_f1 = f1_score(y, y_pred, average="macro", zero_division=0)


    return {
        'name':'Base Model',
        'y':y,
        'y_pred': y_pred,
        'scaling': None,
        'cv':None
    }, base_model_recall, base_model_precision, base_model_f1


    
    # print(base_model_precision)
    # print(base_model_recall)
    # print(base_model_f1)

    # ws.append([f'Base model','', '', '', base_model_precision, 0, base_model_recall, 0, base_model_f1, 0])
    # model_number += 1

def add_model_report_row(
    *,
    ws,
    model_name,
    y_test,
    y_pred,
    label_names,
    # X_train,
    base_model_precision,
    base_model_recall,
    base_model_f1,
    diagrams_dir,
    scaling,
    cv=None,
    export_to_file=True
):
    """
    Adds a single model report row to an Excel worksheet.
    """

    # ========================
    # Confusion matrix
    # ========================
    if export_to_file:
        cm_path = f"{diagrams_dir}/{model_name}_confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(14, 14), dpi=300)  # 👈 high DPI

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=label_names,
            xticks_rotation=90,
            ax=ax
        )

        # Increase number font size
        for text in disp.text_.ravel():
            text.set_fontsize(9)

        # Increase tick label size
        ax.tick_params(axis='both', labelsize=10)

        # Keep cells square
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.cla()

        row_num = ws.max_row + 1

        img = openpyxl.drawing.image.Image(cm_path)
        img.width = int(img.width / 7)
        img.height = int(img.height / 7)
        
        cell_ref = f"K{row_num}"

        ws.column_dimensions["K"].width = img.width/7
        ws.row_dimensions[row_num].height = img.height

        img.anchor = cell_ref
        ws.add_image(img, cell_ref)

    # ========================
    # Metrics
    # ========================
    try:
        scores = classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            output_dict=True
        )
    except Exception as e:
        print("y_test shape:", y_test.shape)
        print("y_pred shape:", y_pred.shape)
        print("Last y_test:", y_test[-10:])
        print("Last y_pred:", y_pred[-10:])
        raise e

    macro_p = scores["macro avg"]["precision"]
    macro_r = scores["macro avg"]["recall"]
    macro_f1 = scores["macro avg"]["f1-score"]

    # ========================
    # Params (if CV model)
    # ========================
    params = ""
    if cv is not None and hasattr(cv, "best_params_"):
        params = str(cv.best_params_)

    # ========================
    # Append row
    # ========================
    ws.append([
        f"{model_name}",
        scaling,
        "",
        params,
        macro_p,
        macro_p / base_model_precision * 100 - 100,
        macro_r,
        macro_r / base_model_recall * 100 - 100,
        macro_f1,
        macro_f1 / base_model_f1 * 100 - 100
    ])




