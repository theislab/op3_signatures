# Change history
# * Imported code from https://www.kaggle.com/code/jankowalski2000/3rd-place-solution?scriptVersionId=153045206
# * Derive gene_names from train_df to to avoid reading in the sample_submission.csv file
# * Make reps component arguments
# * Auto-reformatted the code
# * Restructured the code into a function `run_notebook_264()`

# -----------------------------------------------------------------------------
# Load dependencies
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Embedding,
    Flatten,
    Concatenate,
    Input,
    Lambda,
)
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers.legacy import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD


# -----------------------------------------------------------------------------
# Define helper functions and parameters
# -----------------------------------------------------------------------------
EMBEDDING_COL = {
    "fp": "ECFP:2",
    "lpm": "LPM_emb",
    "pca_logfc": "PCA.logFC",
    "pca_t": "PCA.t",
}


def sm_to_embeddings(compounds_list, emb_df, embedding_col, fp_dim=None):
    """Look up the precomputed fp/lpm vector for each compound.

    Compounds that are missing from ``emb_df`` or whose entry is ``None`` /
    a scalar ``NaN`` are returned as a zero vector of length ``fp_dim`` so
    downstream consumers (model inputs) get a uniformly-shaped placeholder.
    The accompanying per-sample / per-compound mask – built by
    ``build_fp_weights_and_mask`` – is what actually tells the model which
    rows are real and which are placeholders.
    """
    key_col = "sm_name" if "sm_name" in emb_df.columns else "perturbagen"

    if fp_dim is None:
        valid = emb_df.loc[emb_df[embedding_col].notna(), embedding_col]
        fp_dim = len(np.array(valid.iloc[0]))

    zeros = np.zeros(fp_dim, dtype=np.float32)
    rows = []
    for sm in compounds_list:
        match = emb_df.loc[emb_df[key_col] == sm, embedding_col]
        if len(match) == 0:
            rows.append(zeros)
            continue
        val = match.values[0]
        if val is None or (np.ndim(val) == 0 and pd.isna(val)):
            rows.append(zeros)
            continue
        arr = np.array(val, dtype=np.float32)
        if arr.shape == ():
            rows.append(zeros)
            continue
        rows.append(arr)
    return np.stack(rows)


def custom_mean_rowwise_rmse(y_true, y_pred):
    rmse_per_row = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))
    mean_rmse = tf.reduce_mean(rmse_per_row)
    return mean_rmse


def _make_inputs(emb_out, embedding_layer="concat", fp_dim=None, use_fp_dense=False,
                 fp_weights=None, fp_mask=None):
    """Build the shared input head. Returns (inputs, x) where x feeds into the body.

    concat    – cell-type and compound get separate learnable
                ``Embedding(152, emb_out)`` tables. When a precomputed fp/lpm
                vector is available for a compound, ``[sm_emb, fp_proj]`` is
                concatenated and projected back down to ``emb_out`` via a
                ``Dense(emb_out)`` layer (``sm_proj``). For compounds with
                missing fp/lpm, the bare ``sm_emb`` is used instead. The two
                branches are selected per-sample via a frozen ``fp_mask``
                lookup. The result is finally concatenated with ``cell_emb``,
                yielding a ``(batch, 2 * emb_out)`` tensor.
    fixed     – compound Embedding initialised from fp/lpm weights and frozen;
                a Dense projection to ``emb_out`` is learned. Cell-type
                embedding stays fully learnable. Compounds with MISSING fp/lpm
                fall back to a separate learnable ``Embedding(152, emb_out)``
                gated in the same way as the concat branch.
    trainable – same as fixed but the fp-initialised Embedding weights are
                also trainable.

    ``fp_mask`` is a ``(152,)`` float32 array (1.0 for compounds with a
    precomputed embedding, 0.0 for missing). When omitted, it is derived
    from ``fp_weights`` row sums in the fixed/trainable branches; for the
    concat branch with ``fp_dim`` set, omitting it disables masking and
    every compound is treated as having an embedding (back-compat).
    """
    cat_input = Input(shape=(2,), name="cat_input")

    cell_idx = Lambda(lambda z: z[:, 0:1], name="cell_idx")(cat_input)
    sm_idx   = Lambda(lambda z: z[:, 1:2], name="sm_idx")(cat_input)

    if embedding_layer == "concat":
        # Separate learnable embeddings for cell type and compound (rather
        # than the original shared 152-vocab table).
        cell_emb = Embedding(152, emb_out, name="cell_embedding")(cell_idx)
        cell_emb = Flatten()(cell_emb)

        sm_emb = Embedding(152, emb_out, name="sm_embedding")(sm_idx)
        sm_emb = Flatten()(sm_emb)

        if fp_dim is None:
            # No external fp at all; just stack the two embeddings.
            x = Concatenate()([cell_emb, sm_emb])
            return cat_input, x

        fp_input = Input(shape=(fp_dim,), name="fp_input")
        x2 = Dense(256)(fp_input) if use_fp_dense else fp_input

        # Has-fp branch: concat sm embedding with fp features and project
        # back down to emb_out so both branches share the same dimensionality.
        sm_combined = Concatenate()([sm_emb, x2])
        sm_proj = Dense(emb_out, name="sm_projection")(sm_combined)

        if fp_mask is None:
            # No mask provided -> every compound treated as having fp.
            sm_final = sm_proj
        else:
            mask_w = np.asarray(fp_mask, dtype=np.float32).reshape(152, 1)
            mask = Embedding(
                152, 1, weights=[mask_w], trainable=False,
                name="fp_mask",
            )(sm_idx)
            mask = Flatten()(mask)
            # mask = 1 -> sm_proj (compound has fp), mask = 0 -> sm_emb (missing).
            sm_final = Lambda(
                lambda t: t[2] * t[0] + (1.0 - t[2]) * t[1],
                name="sm_select",
            )([sm_proj, sm_emb, mask])

        x = Concatenate()([cell_emb, sm_final])
        return [cat_input, fp_input], x

    # ------------------------------------------------------------------
    # fixed / trainable
    # ------------------------------------------------------------------
    cell_emb = Embedding(152, emb_out, name="cell_embedding")(cell_idx)
    cell_emb = Flatten()(cell_emb)

    # compound branch A: Embedding initialised from fp/lpm features -> Dense projection.
    # Used for compounds that DO have a precomputed embedding.
    is_trainable = (embedding_layer == "trainable")
    init_fp_dim = fp_weights.shape[1]
    sm_raw = Embedding(
        152, init_fp_dim,
        weights=[fp_weights], trainable=is_trainable,
        name="sm_embedding_fp",
    )(sm_idx)
    sm_raw = Flatten()(sm_raw)
    sm_proj = Dense(emb_out, name="sm_projection")(sm_raw)

    # compound branch B: standard learnable embedding (initialised from scratch),
    # mirroring the cell-type branch. Used for compounds with MISSING fp/lpm.
    sm_emb = Embedding(152, emb_out, name="sm_embedding_learn")(sm_idx)
    sm_emb = Flatten()(sm_emb)

    if fp_mask is None:
        # Fall back to deriving the mask from fp_weights (zero rows = missing).
        fp_mask = (np.abs(fp_weights).sum(axis=1) != 0).astype(np.float32)
    mask_w = np.asarray(fp_mask, dtype=np.float32).reshape(152, 1)
    mask = Embedding(
        152, 1, weights=[mask_w], trainable=False,
        name="fp_mask",
    )(sm_idx)
    mask = Flatten()(mask)

    # mask = 1 -> sm_proj, mask = 0 -> sm_emb.
    sm_final = Lambda(
        lambda t: t[2] * t[0] + (1.0 - t[2]) * t[1],
        name="sm_select",
    )([sm_proj, sm_emb, mask])

    x = Concatenate()([cell_emb, sm_final])
    return cat_input, x


def build_fp_weights_and_mask(le, emb_df, embedding_col):
    """Build a (152, fp_dim) float32 weight matrix AND a (152,) float32 mask
    for the compound Embedding layer.

    A row at index ``idx`` is considered "valid" (mask = 1, weights filled
    with the fp/lpm vector) iff:

      * the corresponding compound is present in ``emb_df``;
      * the embedding cell is not ``None`` and not a scalar ``NaN``;
      * after array-coercion, the result has non-zero rank.

    All other rows (cell types, compounds absent from ``emb_df``, or
    compounds with ``None`` / scalar-``NaN`` entries) are left as zeros and
    have mask = 0. The mask is used by ``_make_inputs`` to gate the
    fp-based projection vs the from-scratch learnable Embedding fallback.
    """
    valid_col = emb_df.loc[emb_df[embedding_col].notna(), embedding_col]
    fp_dim = len(np.array(valid_col.iloc[0]))

    fp_weights = np.zeros((152, fp_dim), dtype=np.float32)
    fp_mask = np.zeros((152,), dtype=np.float32)
    for sm in emb_df["perturbagen"].unique():
        if sm not in le.classes_:
            continue
        val = emb_df.loc[emb_df["perturbagen"] == sm, embedding_col].values[0]
        if val is None or (np.ndim(val) == 0 and pd.isna(val)):
            continue
        arr = np.array(val, dtype=np.float32)
        if arr.shape == ():
            continue
        idx = int(le.transform([sm])[0])
        fp_weights[idx] = arr
        fp_mask[idx] = 1.0
    return fp_weights, fp_mask


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def model_1(lr, emb_out, n_dim, fp_dim=None, use_fp_dense=False, embedding_layer="concat", fp_weights=None, fp_mask=None):
    inputs, x = _make_inputs(emb_out, embedding_layer, fp_dim, use_fp_dense, fp_weights, fp_mask)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(n_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_2(lr, emb_out, dense_1, dense_2, dropout_1, dropout_2, n_dim, fp_dim=None, use_fp_dense=False, embedding_layer="concat", fp_weights=None, fp_mask=None):
    inputs, x = _make_inputs(emb_out, embedding_layer, fp_dim, use_fp_dense, fp_weights, fp_mask)

    x = Dense(dense_1)(x)  # 64 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_1)(x)  # 256 - 2048
    x = Dense(dense_2, activation="relu")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_2)(x)
    x = Dense(n_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_3(
    lr,
    emb_out,
    dense_1,
    dense_2,
    dense_3,
    dense_4,
    dropout_1,
    dropout_2,
    dropout_3,
    dropout_4,
    n_dim,
    fp_dim=None,
    use_fp_dense=False,
    embedding_layer="concat",
    fp_weights=None,
    fp_mask=None,
):
    inputs, x = _make_inputs(emb_out, embedding_layer, fp_dim, use_fp_dense, fp_weights, fp_mask)

    x = Dense(dense_1)(x)  # 128 - 1024
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_1)(x)
    x = Dense(dense_2)(x)  # 64 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_2)(x)
    x = Dense(dense_3)(x)  # 32 - 256
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_3)(x)
    x = Dense(dense_4)(x)  # 16 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_4)(x)
    x = Dense(n_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_4(
    lr, emb_out, dense_1, dense_2, dense_3, dropout_1, dropout_2, dropout_3, n_dim,
    fp_dim=None, use_fp_dense=False, embedding_layer="concat", fp_weights=None, fp_mask=None,
):
    inputs, x = _make_inputs(emb_out, embedding_layer, fp_dim, use_fp_dense, fp_weights, fp_mask)

    x = Dense(dense_1)(x)  # 128 - 1024
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_1)(x)
    x = Dense(dense_2)(x)  # 64 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_2)(x)
    x = Dense(dense_3)(x)  # 32 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_3)(x)
    x = Dense(n_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss="mae", optimizer=Adam(learning_rate=lr), metrics=[custom_mean_rowwise_rmse]
    )
    return model


def model_5(lr, emb_out, n_dim, dropout_1, dropout_2, fp_dim=None, use_fp_dense=False, embedding_layer="concat", fp_weights=None, fp_mask=None):
    inputs, x = _make_inputs(emb_out, embedding_layer, fp_dim, use_fp_dense, fp_weights, fp_mask)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_1)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_2)(x)
    x = Dense(n_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_6(lr, emb_out, dense_1, dense_2, n_dim, dropout_1, dropout_2, fp_dim=None, use_fp_dense=False, embedding_layer="concat", fp_weights=None, fp_mask=None):
    inputs, x = _make_inputs(emb_out, embedding_layer, fp_dim, use_fp_dense, fp_weights, fp_mask)

    x = BatchNormalization()(x)
    x = Dense(dense_1)(x)  # 64 - 512
    x = Activation("relu")(x)
    x = Dropout(dropout_2)(x)
    x = Dense(dense_2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_2)(x)
    x = Dense(n_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_7(
    lr,
    emb_out,
    dense_1,
    dense_2,
    dense_3,
    dense_4,
    dropout_1,
    dropout_2,
    dropout_3,
    dropout_4,
    n_dim,
    fp_dim=None,
    use_fp_dense=False,
    embedding_layer="concat",
    fp_weights=None,
    fp_mask=None,
):
    inputs, x = _make_inputs(emb_out, embedding_layer, fp_dim, use_fp_dense, fp_weights, fp_mask)

    x = Dense(dense_1)(x)  # 128 - 1024
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_1)(x)
    x = Dense(dense_2)(x)  # 64 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_2)(x)
    x = Dense(dense_3)(x)  # 32 - 256
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_3)(x)
    x = Dense(dense_4)(x)  # 16 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_4)(x)
    x = Dense(n_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def model_8(
    lr, emb_out, dense_1, dense_2, dense_3, dropout_1, dropout_2, dropout_3, n_dim,
    fp_dim=None, use_fp_dense=False, embedding_layer="concat", fp_weights=None, fp_mask=None,
):
    inputs, x = _make_inputs(emb_out, embedding_layer, fp_dim, use_fp_dense, fp_weights, fp_mask)

    x = Dense(dense_1)(x)  # 128 - 1024
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_1)(x)
    x = Dense(dense_2)(x)  # 64 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_2)(x)
    x = Dense(dense_3)(x)  # 32 - 512
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_3)(x)
    x = Dense(n_dim, activation="linear")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(
        loss=custom_mean_rowwise_rmse,
        optimizer=Adam(learning_rate=lr),
        metrics=[custom_mean_rowwise_rmse],
    )
    return model


def load_models():
    models = [
        model_1,
        model_2,
        model_3,
        model_5,
        model_6,
        model_7,
        model_8,
    ]
    return models


# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------
def load_params():
    params_model_1 = {
        "params": {
            "epochs": 114,
            "bs": 128,
            "lr": 0.008457844054540857,
            "emb_out": 22,
            "n_dim": 50,
        },
        "value": 0.9060678655727635,
    }

    params_model_2 = {
        "params": {
            "epochs": 136,
            "bs": 64,
            "lr": 0.007787474024659863,
            "emb_out": 10,
            "dense_1": 384,
            "dense_2": 1280,
            "dropout_1": 0.4643149193312417,
            "dropout_2": 0.10101884612160547,
            "n_dim": 60,
        },
        "value": 0.9070240468092804,
    }

    params_model_3 = {
        "params": {
            "epochs": 157,
            "bs": 64,
            "lr": 0.004311857150745656,
            "emb_out": 62,
            "dense_1": 560,
            "dense_2": 480,
            "dense_3": 248,
            "dense_4": 224,
            "dropout_1": 0.4359908049836846,
            "dropout_2": 0.34432694543970555,
            "dropout_3": 0.01112409967333259,
            "dropout_4": 0.23133616975077548,
            "n_dim": 119,
        },
        "value": 0.9171315640806535,
    }

    params_model_4 = {
        "params": {
            "epochs": 147,
            "bs": 64,
            "lr": 0.005948541271442179,
            "emb_out": 46,
            "dense_1": 872,
            "dense_2": 264,
            "dense_3": 256,
            "dropout_1": 0.17543603718794346,
            "dropout_2": 0.3587657616370447,
            "dropout_3": 0.12077512068514727,
            "n_dim": 213,
        },
        "value": 0.9228638968500431,
    }

    params_model_5 = {
        "params": {
            "epochs": 122,
            "bs": 32,
            "lr": 0.004429076555977599,
            "emb_out": 32,
            "n_dim": 71,
            "dropout_1": 0.40604535344002984,
            "dropout_2": 0.178189970426619,
        },
        "value": 0.9083640103276015,
    }

    params_model_6 = {
        "params": {
            "epochs": 112,
            "bs": 128,
            "lr": 0.009773732221901085,
            "emb_out": 60,
            "dense_1": 436,
            "dense_2": 416,
            "n_dim": 126,
            "dropout_1": 0.4024659444883379,
            "dropout_2": 0.2573940194596736,
        },
        "value": 0.8909352668212382,
    }

    params_model_7 = {
        "params": {
            "epochs": 141,
            "bs": 128,
            "lr": 0.005530331519967936,
            "emb_out": 48,
            "dense_1": 712,
            "dense_2": 400,
            "dense_3": 232,
            "dense_4": 216,
            "dropout_1": 0.4903998136177629,
            "dropout_2": 0.032371643764537134,
            "dropout_3": 0.11138300987168903,
            "dropout_4": 0.019885384663655765,
            "n_dim": 100,
        },
        "value": 0.8978272722102707,
    }

    params_model_8 = {
        "params": {
            "epochs": 143,
            "bs": 192,
            "lr": 0.00971858172843266,
            "emb_out": 48,
            "dense_1": 312,
            "dense_2": 344,
            "dense_3": 248,
            "dropout_1": 0.10974777738609129,
            "dropout_2": 0.10106027333885811,
            "dropout_3": 0.09775833250663657,
            "n_dim": 100,
        },
        "value": 0.8885448573595669,
    }

    params = [
        params_model_1,
        params_model_2,
        params_model_3,
        params_model_5,
        params_model_6,
        params_model_7,
        params_model_8,
    ]
    return params


# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------
def load_weights():
    w1 = [
        0.15224443321212433,
        0.7152220796128623,
        0.7547606691460997,
        0.05786285275052854,
        0.9602177109190158,
        0.4968056740470425,
        0.9881673272809887,
    ]
    return w1


# -----------------------------------------------------------------------------
# Predict functions
# -----------------------------------------------------------------------------
def split_params_to_training_model(model_params, embedding_type, embedding_layer, fp_dim, use_fp_dense):
    model_params = model_params["params"]
    training_keys = ["epochs", "bs"]
    training_params = {k: model_params[k] for k in training_keys}
    model_params = {
        k: model_params[k] for k in model_params.keys() if k not in training_keys
    }
    if embedding_type != "none":
        model_params["embedding_layer"] = embedding_layer
        if embedding_layer == "concat":
            model_params["fp_dim"] = fp_dim
            model_params["use_fp_dense"] = use_fp_dense
    return model_params, training_params


def fit_and_predict_embedding_nn(
    x, y, test_x, model_constructor, best_params,
    embedding_type, embedding_layer, fp_dim, use_fp_dense,
    train_fps=None, fp_test=None, fp_weights=None, fp_mask=None,
):
    model_params, training_params = split_params_to_training_model(
        best_params, embedding_type, embedding_layer, fp_dim, use_fp_dense
    )
    if embedding_type != "none":
        if embedding_layer != "concat":
            model_params["fp_weights"] = fp_weights
        if fp_mask is not None:
            model_params["fp_mask"] = fp_mask
    n_dim = model_params["n_dim"]
    d = TruncatedSVD(n_dim)
    y = d.fit_transform(y)
    model = model_constructor(**model_params)
    use_separate_fp = (embedding_type != "none" and embedding_layer == "concat")
    x_in      = [x, train_fps] if use_separate_fp else x
    test_x_in = [test_x, fp_test] if use_separate_fp else test_x
    model.fit(
        x_in,
        y,
        epochs=training_params["epochs"],
        batch_size=training_params["bs"],
        verbose=0,
        shuffle=True,
    )
    return d.inverse_transform(model.predict(test_x_in, batch_size=1))


def predict(
    test_df, models, params, weights, le, new_names, original_y, reps,
    emb_df, embedding_type, embedding_layer, fp_dim, use_fp_dense,
    train_fps=None, fp_weights=None, fp_mask=None,
):
    x_test = le.transform(test_df[["cell_type", "sm_name"]].values.flat).reshape(-1, 2)

    fp_test = None
    if embedding_type != "none" and embedding_layer == "concat":
        embedding_col = EMBEDDING_COL[embedding_type]
        fp_test = sm_to_embeddings(test_df["sm_name"], emb_df, embedding_col, fp_dim=fp_dim)

    preds = []
    for model_i in range(len(models)):
        model = models[model_i]
        param = params[model_i]
        temp_pred = []
        for rep_i in range(reps):
            print(
                f"NB264, Training model {model_i + 1}/{len(models)}, Repeat {rep_i + 1}/{reps}",
                flush=True,
            )
            temp_pred.append(
                fit_and_predict_embedding_nn(
                    new_names, original_y, x_test, model, param,
                    embedding_type, embedding_layer, fp_dim, use_fp_dense,
                    train_fps, fp_test, fp_weights, fp_mask,
                )
            )
        temp_pred = np.median(temp_pred, axis=0)
        preds.append(temp_pred)

    pred = np.sum([w * p for w, p in zip(weights, preds)], axis=0) / sum(weights)
    return pred


def run_notebook_264(train_df, test_df, gene_names, reps, emb_df, embedding_type, use_fp_dense=True, embedding_layer="concat", seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # determine mins and maxs for later clipping
    original_y = train_df.loc[:, gene_names].values
    mins = original_y.min(axis=0)
    maxs = original_y.max(axis=0)

    # determine label encoder
    original_x = train_df[["cell_type", "sm_name"]].values
    all_x = pd.concat([train_df[["cell_type", "sm_name"]], test_df[["cell_type", "sm_name"]]]).values
    le = LabelEncoder()
    le.fit(all_x.flat)
    new_names = le.transform(original_x.flat).reshape(-1, 2)

    # compute external embeddings for train if needed
    train_fps = None
    fp_dim = None
    fp_weights = None
    fp_mask = None
    if embedding_type != "none":
        embedding_col = EMBEDDING_COL[embedding_type]
        # build_fp_weights_and_mask gives us a uniform "missing compound"
        # definition for all three embedding_layer modes; the concat branch
        # only consumes fp_mask, fixed/trainable consume both.
        fp_weights, fp_mask = build_fp_weights_and_mask(le, emb_df, embedding_col)
        if embedding_layer == "concat":
            fp_dim = fp_weights.shape[1]
            train_fps = sm_to_embeddings(train_df["sm_name"], emb_df, embedding_col, fp_dim=fp_dim)
            fp_weights = None  # concat doesn't consume the weight matrix

    # load models, params, and weights
    models = load_models()
    params = load_params()
    weights = load_weights()

    # generate predictions
    pred = predict(
        test_df, models, params, weights, le, new_names, original_y, reps,
        emb_df, embedding_type, embedding_layer, fp_dim, use_fp_dense,
        train_fps, fp_weights, fp_mask,
    )

    # clip predictions
    clipped_pred = np.clip(pred, mins, maxs)

    # format outputs
    df = pd.DataFrame(clipped_pred, columns=gene_names)
    df["id"] = range(len(df))
    df = df.loc[:, ["id"] + gene_names]

    return df
