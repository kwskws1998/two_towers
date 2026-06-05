import json
import pandas as pd
import csv
import numpy as np
import torch
import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr as scipy_pearsonr

preds_dir = None


def set_preds_dir(path):
    global preds_dir
    preds_dir = path


# This function handles CTRL-L C interrupt, erasing unused folders and terminating the program
def handle_signal(signum, stackframe):
    ''' signal handler '''
    # Best-effort cleanup of the output folder if it is still empty.
    if preds_dir and os.path.isdir(preds_dir):
        try:
            os.rmdir(preds_dir)
        except OSError:
            pass
    print('\n')
    sys.exit(1)


def _safe_pearson_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(scipy_pearsonr(y_true, y_pred)[0])


def _calculate_va_metrics(df):
    metrics = {"num_samples": int(len(df))}
    for dim in ["valence", "arousal"]:
        y_true = df[f"{dim}_true"]
        y_pred = df[f"{dim}_pred"]
        mse = float(mean_squared_error(y_true, y_pred))
        metrics[f"mse_{dim}"] = mse
        metrics[f"rmse_{dim}"] = float(np.sqrt(mse))
        metrics[f"mae_{dim}"] = float(mean_absolute_error(y_true, y_pred))
        metrics[f"pearson_corr_{dim}"] = _safe_pearson_corr(y_true, y_pred)
    return metrics


def _json_safe_metrics(metrics):
    safe = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        safe[key] = value
    return safe


def _write_out_of_fold_metrics(path, df_join):
    overall_metrics = _calculate_va_metrics(df_join)
    pd.DataFrame([overall_metrics]).to_csv(path + "/overall_metrics.csv", index=False)
    with open(path + "/overall_metrics.json", "w") as output_file:
        json.dump(_json_safe_metrics(overall_metrics), output_file, indent=2)

    dataset_rows = []
    for dataset_name, df_dataset in df_join.groupby("dataset_of_origin"):
        row = {"dataset_of_origin": dataset_name}
        row.update(_calculate_va_metrics(df_dataset))
        dataset_rows.append(row)
    pd.DataFrame(dataset_rows).sort_values("dataset_of_origin").to_csv(
        path + "/dataset_metrics.csv", index=False
    )


def _join_dataset_and_predictions(dataset_df, predictions_df, prediction_filename):
    if len(dataset_df) != len(predictions_df):
        raise ValueError(
            f"{prediction_filename} has {len(predictions_df)} rows, "
            f"but the matching dataset fold has {len(dataset_df)} rows. "
            "Use the same data/full_dataset_fold*.csv files that were used during training."
        )
    return pd.concat(
        [dataset_df.reset_index(drop=True), predictions_df.reset_index(drop=True)],
        axis=1,
    )


# Code tested on script_sort_predictions_temp.ipynb
def create_prediction_tables(path, data_dir="data"):
    df_preds_fold1 = pd.read_csv(path + '/predictions_fold1.csv')
    df_preds_fold2 = pd.read_csv(path + '/predictions_fold2.csv')
    df_preds_fold1 = df_preds_fold1.rename(columns={'Unnamed: 0' : 'index_pred', '0' : 'valence_pred', '1' : 'arousal_pred'})
    df_preds_fold2 = df_preds_fold2.rename(columns={'Unnamed: 0' : 'index_pred', '0' : 'valence_pred', '1' : 'arousal_pred'})
    
    # Import original dataset files to df
    df_dataset_fold1 = pd.read_csv(os.path.join(data_dir, 'full_dataset_fold1.csv'),sep='\t',
                    quotechar='"',
                    engine='python', 
                    quoting=csv.QUOTE_NONE,
                    escapechar='\\',
                    keep_default_na=False,
                    dtype={'index':np.int32,'text':str,'valence':np.float64, 'arousal':np.float64})
    df_dataset_fold2 = pd.read_csv(os.path.join(data_dir, 'full_dataset_fold2.csv'),sep='\t',
                    quotechar='"',
                    engine='python', 
                    quoting=csv.QUOTE_NONE,
                    escapechar='\\',
                    keep_default_na=False,
                    dtype={'index':np.int32,'text':str,'valence':np.float64, 'arousal':np.float64})
    df_dataset_fold1 = df_dataset_fold1.rename(columns={'valence' : 'valence_true', 'arousal' : 'arousal_true'})
    df_dataset_fold2 = df_dataset_fold2.rename(columns={'valence' : 'valence_true', 'arousal' : 'arousal_true'})

    # Merge the original dataset and the predicted values in the same dataframe

    # Fold 1
    df_fold1_join = _join_dataset_and_predictions(
        df_dataset_fold1, df_preds_fold1, "predictions_fold1.csv"
    )
    df_fold1_join = df_fold1_join.drop(columns=['index_pred']) # Drop extra index column
    cols = df_fold1_join.columns.tolist() # Re-order columns
    cols = ['index', 'text', 'dataset_of_origin', 'valence_true', 'arousal_true', 'valence_pred', 'arousal_pred']
    df_fold1_join = df_fold1_join[cols]

    # Fold 2
    df_fold2_join = _join_dataset_and_predictions(
        df_dataset_fold2, df_preds_fold2, "predictions_fold2.csv"
    )
    df_fold2_join = df_fold2_join.drop(columns=['index_pred']) # Drop extra index column
    cols = df_fold2_join.columns.tolist() # Re-order columns
    cols = ['index', 'text', 'dataset_of_origin', 'valence_true', 'arousal_true', 'valence_pred', 'arousal_pred']
    df_fold2_join = df_fold2_join[cols]
    
    df_join = pd.concat([df_fold1_join, df_fold2_join], axis=0)

    # Sort dataframe by index
    df_join = df_join.sort_values('index')
    df_join.to_csv(path + "/all_predictions.csv", index=False)
    _write_out_of_fold_metrics(path, df_join)
    
    # A list with the name of all the datasets used 
    datasets_list = list(df_join.dataset_of_origin.unique())
    # len(datasets_list) # 33 datasets

    # words  dataset
    words_ds_list = ['ANEW to EP', 'ANGST', 'ANPW_R', 'BAWL_R', 'Cantonese Nouns','Chinese words', 'ChineseW11k', 'CroatianNorms', 'DutchAdj', 'FAN - french words', 'FEEL', 'FinnishNorms', 'FinnishNouns', 'German words', 'GlasgowNorms', 
    'Italian words', 'NAWL', 'nrc-vad', 'TurkishNorms', 'word ratings NL', 'word ratings ES', 'word ratings ENG']
    # sentences dataset
    sent_ds_list = ['ANET sentences', 'CVAI', 'CVAT', 'COMETA sentences', 'COMETA stories', 'Emobank', 'EmoTales sentences', 'fb', 'IEMOCAP sentences', 'MAS', 'PANIG sentences', 'Polish sentences']

    dataset_langs = {
        'ANGST': "German", 'BAWL_R': "German", 'German words': "German", 'COMETA sentences': "German", 'COMETA stories': "German", 'PANIG sentences': "German",
        'ANPW_R' : "Polish", 'NAWL' : "Polish", 'Polish sentences' : "Polish", 'Chinese words' : "Mandarin", 'ChineseW11k' : "Mandarin", 'CVAI' : "Mandarin",
        'CVAT' : "Mandarin", 'FAN - french words' : "French", 'FEEL' : "French", 'Italian words' : "Italian", 'CroatianNorms' : "Croatian", 'FinnishNorms' : 'Finnish',
        'FinnishNouns' : 'Finnish', 'TurkishNorms' : 'Turkish', 'word ratings NL' : "Dutch", 'DutchAdj' : "Dutch", 'GlasgowNorms' : 'English', 'nrc-vad' : 'English',
        'word ratings ENG' : 'English', 'ANET sentences' : 'English', 'Emobank' : 'English',  'EmoTales sentences' : 'English', 'fb' : 'English', 'IEMOCAP sentences' : 'English',
        'word ratings ES' : 'Spanish', 'Cantonese Nouns' : 'Cantonese', 'ANEW to EP' : 'Portuguese', 'MAS' : 'Portuguese'
    }

    # Keep only datasets available in the current run (e.g., English-only subsets).
    available_datasets = set(df_join.dataset_of_origin.unique())
    words_ds_list = [ds for ds in words_ds_list if ds in available_datasets]
    sent_ds_list = [ds for ds in sent_ds_list if ds in available_datasets]

    # add ds_type column
    temp_word = df_join[df_join['dataset_of_origin'].isin(words_ds_list)] #['ds_type'] = 'word'
    temp_word = temp_word.assign(ds_type = 'word')
    temp_sent = df_join[df_join['dataset_of_origin'].isin(sent_ds_list)] #['ds_type'] = 'word'
    temp_sent = temp_sent.assign(ds_type = 'sentence')
    full_df = pd.concat([temp_word, temp_sent], axis=0)
    
    # add language column
    german = ['ANGST', 'BAWL_R','German words', 'COMETA sentences', 'COMETA stories', 'PANIG sentences']
    polish = ['ANPW_R','NAWL', 'Polish sentences']
    mandarin = ['Chinese words','ChineseW11k','CVAI','CVAT']
    french = ['FAN - french words','FEEL']
    italian = ['Italian words']
    croatian = ['CroatianNorms']
    finnish = ['FinnishNorms','FinnishNouns']
    turkish = ['TurkishNorms']
    dutch = ['word ratings NL','DutchAdj']
    english = ['GlasgowNorms','nrc-vad','word ratings ENG','ANET sentences','Emobank', 'EmoTales sentences', 'fb', 'IEMOCAP sentences']
    spanish = ['word ratings ES']
    cantonese = ['Cantonese Nouns']
    portuguese = ['ANEW to EP','MAS']
    
    # Add columns language and type
    def add_column_lang(ds_origin):
        return dataset_langs.get(ds_origin, 'Unknown')

    # run add col lang function
    full_df['language'] = full_df.dataset_of_origin.apply(add_column_lang)
    
    # TABLES 1 - Word datasets

    # Array containing languages to fill df
    lang_array = []
    mse_val_array = []
    mse_aro_array = []
    mae_val_array = []
    mae_aro_array = []
    r_val_array = []
    r_aro_array = []

    if words_ds_list:
        for ds in words_ds_list:
            # language
            l = dataset_langs.get(ds, 'Unknown')
            lang_array.append(l)
            df_temp = full_df[full_df.dataset_of_origin == ds]
            # How to calculate MSE, MAE for valence and arousal for one of the datasets
            mse_valence = np.sqrt(mean_squared_error(df_temp.valence_true, df_temp.valence_pred))
            mse_arousal = np.sqrt(mean_squared_error(df_temp.arousal_true, df_temp.arousal_pred))
            mae_valence = mean_absolute_error(df_temp.valence_true, df_temp.valence_pred)
            mae_arousal = mean_absolute_error(df_temp.arousal_true, df_temp.arousal_pred)
            r_valence = scipy_pearsonr(df_temp.valence_true, df_temp.valence_pred)
            r_arousal = scipy_pearsonr(df_temp.arousal_true, df_temp.arousal_pred)
           
            # Append values to its arrays
            mse_val_array.append(round(mse_valence,4))
            mse_aro_array.append(round(mse_arousal,4))
            mae_val_array.append(round(mae_valence,4))
            mae_aro_array.append(round(mae_arousal,4))
            r_val_array.append(round(r_valence[0],4))
            r_aro_array.append(round(r_arousal[0],4))

        # Arrays to put in the df
        ds_array = np.array(words_ds_list).reshape(len(words_ds_list), 1)
        lang_array = np.array(lang_array).reshape(len(lang_array), 1)
        mse_val_array = np.array(mse_val_array).reshape(len(mse_val_array), 1)
        mse_aro_array = np.array(mse_aro_array).reshape(len(mse_aro_array), 1)
        mae_val_array = np.array(mae_val_array).reshape(len(mae_val_array), 1)
        mae_aro_array = np.array(mae_aro_array).reshape(len(mae_aro_array), 1)
        r_val_array = np.array(r_val_array).reshape(len(r_val_array), 1)
        r_aro_array = np.array(r_aro_array).reshape(len(r_aro_array), 1)

        matrix = np.hstack((ds_array, lang_array, mse_val_array, mae_val_array, r_val_array, mse_aro_array, mae_aro_array, r_aro_array))
        # Putting the df together
        header = [np.array(['', '','Valence', 'Valence', 'Valence', 'Arousal', 'Arousal', 'Arousal']), 
        np.array(['Dataset','Language', 'MSE', 'MAE', 'r', 'MSE', 'MAE', 'r'])]

        df = pd.DataFrame(matrix, columns= header) #, index=ind

        def df_style(val):
            return "font-weight: bold"

        v_mse_mean = np.mean(np.array(df.Valence.MSE, dtype=float))
        v_mae_mean = np.mean(np.array(df.Valence.MAE, dtype=float))
        v_r_mean = np.mean(np.array(df.Valence.r, dtype=float))
        a_mse_mean = np.mean(np.array(df.Arousal.MSE, dtype=float))
        a_mae_mean = np.mean(np.array(df.Arousal.MAE, dtype=float))
        a_r_mean = np.mean(np.array(df.Arousal.r, dtype=float))
        df.loc[df.shape[0]] = ['Overall','', round(v_mse_mean,4), round(v_mae_mean,4), round(v_r_mean,4), round(a_mse_mean,4), round(a_mae_mean,4), round(a_r_mean,4)]
    else:
        df = pd.DataFrame()
    df.to_pickle(path + "/table1.pkl")

    # TABLE 2 - Sentence datasets
    # Array containing languages to fill df
    lang_array = []
    mse_val_array = []
    mse_aro_array = []
    mae_val_array = []
    mae_aro_array = []
    r_val_array = []
    r_aro_array = []

    if sent_ds_list:
        for ds in sent_ds_list:
            # language
            l = dataset_langs.get(ds, 'Unknown')
            lang_array.append(l)
            #get sub-df
            df_temp = full_df[full_df.dataset_of_origin == ds]
            # How to calculate RMSE, MAE for valence and arousal for one of the datasets
            mse_valence = np.sqrt(mean_squared_error(df_temp.valence_true, df_temp.valence_pred))
            mse_arousal = np.sqrt(mean_squared_error(df_temp.arousal_true, df_temp.arousal_pred))
            mae_valence = mean_absolute_error(df_temp.valence_true, df_temp.valence_pred)
            mae_arousal = mean_absolute_error(df_temp.arousal_true, df_temp.arousal_pred)
            r_valence = scipy_pearsonr(df_temp.valence_true, df_temp.valence_pred)
            r_arousal = scipy_pearsonr(df_temp.arousal_true, df_temp.arousal_pred)
            # Append values to its arrays
            mse_val_array.append(round(mse_valence,4))
            mse_aro_array.append(round(mse_arousal,4))
            mae_val_array.append(round(mae_valence,4))
            mae_aro_array.append(round(mae_arousal,4))
            r_val_array.append(round(r_valence[0],4))
            r_aro_array.append(round(r_arousal[0],4))

        # Arrays to put in the df
        ds_array = np.array(sent_ds_list).reshape(len(sent_ds_list), 1)
        lang_array = np.array(lang_array).reshape(len(lang_array), 1)
        mse_val_array = np.array(mse_val_array).reshape(len(mse_val_array), 1)
        mse_aro_array = np.array(mse_aro_array).reshape(len(mse_aro_array), 1)
        mae_val_array = np.array(mae_val_array).reshape(len(mae_val_array), 1)
        mae_aro_array = np.array(mae_aro_array).reshape(len(mae_aro_array), 1)
        r_val_array = np.array(r_val_array).reshape(len(r_val_array), 1)
        r_aro_array = np.array(r_aro_array).reshape(len(r_aro_array), 1)
        matrix = np.hstack((ds_array, lang_array, mse_val_array, mae_val_array, r_val_array, mse_aro_array, mae_aro_array, r_aro_array))
        # Putting the df together
        header = [np.array(['', '','Valence', 'Valence', 'Valence', 'Arousal', 'Arousal', 'Arousal']), 
        np.array(['Dataset','Language', 'MSE', 'MAE', 'r', 'MSE', 'MAE', 'r'])]

        df = pd.DataFrame(matrix, columns= header) #, index=ind

        v_mse_mean = np.mean(np.array(df.Valence.MSE, dtype=float))
        v_mae_mean = np.mean(np.array(df.Valence.MAE, dtype=float))
        v_r_mean = np.mean(np.array(df.Valence.r, dtype=float))
        a_mse_mean = np.mean(np.array(df.Arousal.MSE, dtype=float))
        a_mae_mean = np.mean(np.array(df.Arousal.MAE, dtype=float))
        a_r_mean = np.mean(np.array(df.Arousal.r, dtype=float))
        df.loc[df.shape[0]] = ['Overall','', round(v_mse_mean,4), round(v_mae_mean,4), round(v_r_mean,4), round(a_mse_mean,4), round(a_mae_mean,4), round(a_r_mean,4)]
    else:
        df = pd.DataFrame()
    df.to_pickle(path + "/table2.pkl")
   
   
   
   
   
def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c
