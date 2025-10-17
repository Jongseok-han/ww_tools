#!/usr/bin/env python3
"""
Xano API wrapper for WisdomOwl company analysis notebook.
"""

import requests
import pandas as pd
from typing import Optional, Dict, List, Union


# Xano API configuration
XANO_INSTANCE_BASEPATH = "https://xqnp-dinh-rvpe.n7e.xano.io/api:ethUmi9m"

# Endpoint URLs
ENDPOINTS = {
    'company': f"{XANO_INSTANCE_BASEPATH}/company",
    'cyclicality': f"{XANO_INSTANCE_BASEPATH}/cyclicality",
    'growth': f"{XANO_INSTANCE_BASEPATH}/growth",
    'valuation': f"{XANO_INSTANCE_BASEPATH}/valuation",
    'forward_pe': f"{XANO_INSTANCE_BASEPATH}/forward_pe",
    'lifecycle': f"{XANO_INSTANCE_BASEPATH}/lifecycle",
    'charts': f"{XANO_INSTANCE_BASEPATH}/charts",
    'cagr_prediction': f"{XANO_INSTANCE_BASEPATH}/cagr_prediction",
    'overview': f"{XANO_INSTANCE_BASEPATH}/overview"
}


def _make_request(url: str, method: str = 'GET', params: Optional[Dict] = None,
                  json_data: Optional[Dict] = None, timeout: int = 30) -> Dict:
    """Make HTTP request to Xano API."""
    headers = {'Content-Type': 'application/json'}
    response = requests.request(
        method=method, url=url, params=params, json=json_data,
        headers=headers, timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def get_all_companies_data(endpoint: str, as_dataframe: bool = True) -> Union[List[Dict], pd.DataFrame]:
    """
    Get data from a specific endpoint for all companies.
    This is the recommended way to fetch data.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Invalid endpoint: {endpoint}. Valid options: {list(ENDPOINTS.keys())}")

    data = _make_request(ENDPOINTS[endpoint])

    if as_dataframe:
        return pd.DataFrame(data if isinstance(data, list) else [data])
    return data


def get_all_data(as_dataframe: bool = True) -> Dict[str, Union[pd.DataFrame, List[Dict]]]:
    """
    Get ALL data from ALL endpoints efficiently (9 API calls total).
    Perfect for notebook workflow.
    """
    return {
        'company': get_all_companies_data('company', as_dataframe),
        'cyclicality': get_all_companies_data('cyclicality', as_dataframe),
        'growth': get_all_companies_data('growth', as_dataframe),
        'valuation': get_all_companies_data('valuation', as_dataframe),
        'forward_pe': get_all_companies_data('forward_pe', as_dataframe),
        'lifecycle': get_all_companies_data('lifecycle', as_dataframe),
        'charts': get_all_companies_data('charts', as_dataframe),
        'cagr_prediction': get_all_companies_data('cagr_prediction', as_dataframe),
        'overview': get_all_companies_data('overview', as_dataframe)
    }


def df_join(df, join_df, suffix, left_key='id', right_key='company_id'):
    if join_df.empty:
        return df

    join_df_suffixed = join_df.add_suffix(f'_{suffix}')
    right_key_suffixed = f'{right_key}_{suffix}'
    
    result = df.merge(
        join_df_suffixed,
        left_on=left_key,
        right_on=right_key_suffixed,
        how='left'
    )

    columns_to_drop = [right_key_suffixed, f'id_{suffix}']    
    columns_to_drop = [col for col in columns_to_drop if col in result.columns]
    if columns_to_drop:
        result = result.drop(columns=columns_to_drop)
    
    return result


def preprocess_string_numbers(df):
    df_converted = df.copy()
    conversion_report = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # skip pre-defined non-numeric columns
            if col.lower() in ['ticker', 'name', 'website', 'image', 'sector', 'industry']:
                continue
                
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_original = df[col].notna().sum()
            non_null_converted = converted.notna().sum()
            
            if non_null_original > 0 and (non_null_converted / non_null_original) > 0.7:
                df_converted[col] = converted
                conversion_report[col] = {
                    'converted_count': non_null_converted,
                    'total_count': non_null_original,
                    'success_rate': f"{(non_null_converted / non_null_original) * 100:.1f}%",
                    'sample_before': list(df[col].dropna().head(3)),
                    'sample_after': list(converted.dropna().head(3))
                }
    
    print(f"📊 Summary: Converted {len(conversion_report)} columns to numeric")
    return df_converted, conversion_report