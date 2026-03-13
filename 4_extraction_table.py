import random
from pathlib import Path

import polars as pl
import pandas as pd

from utils.datasplitter import DataSplitter


class ExtractionTable(DataSplitter):
    """
    This class Produces Table 1. from the article.
    get_flat_table describes the demographics
    get_ts_table describes the continuous variables
    get_unit_table describe the unit types
    get_med_table describes the drug exposure
    """
    def __init__(self):
        super().__init__(equal_samples=True, recompute_index=False)
        random.seed(974)
        self.drug_exposure_savepath = 'figures/main_table/drug_exposures.parquet'
        self.df = pl.scan_parquet(self.extracted_labels_pth)
        
        self.extracted_ts_pths = self._get_extraction_ts_pths()
        
    def get_flat_table(self):
        tab = (self.df
               .group_by('source_dataset')
               .agg(
                   pl.col('original_uniquepid').n_unique().alias('patient'),
                   pl.col('patient').n_unique().alias('stays'),
                   pl.col('raw_age').mean().alias('age'),
                   pl.col('raw_age').std().alias('age_std'),
                   pl.col('sex').eq(0).sum().alias('n_sex_female'),
                   pl.col('sex').eq(0).mean().mul(100).alias('p_sex_female'),
                   pl.col('sex').eq(1).sum().alias('n_sex_male'),
                   pl.col('sex').eq(1).mean().mul(100).alias('p_sex_male'),
                   pl.col('sex').eq(0.5).sum().alias('n_sex_unknown'),
                   pl.col('sex').eq(0.5).mean().mul(100).alias('p_sex_unknown'),
                   pl.col('raw_height').mean().alias('height'),
                   pl.col('raw_height').std().alias('height_std'),
                   pl.col('raw_weight').mean().alias('weight'),
                   pl.col('raw_weight').std().alias('weight_std'),
                   pl.col('lengthofstay').mean().mul(24).alias('LoS'),
                   pl.col('lengthofstay').mul(24).std().alias('LoS_std'),
                   pl.col('lengthofstay').ge(10).sum().alias('n_long_los'),
                   pl.col('lengthofstay').ge(10).mean().mul(100).alias('p_long_los'),
                   pl.col('mortality').mean().mul(100).alias('p_mortality'),
                   pl.col('mortality').ge(1).sum().alias('n_mortality'),
                )
               .filter(pl.col('source_dataset').ne('mimic3'))
               .collect()
               .to_pandas())
        return tab

    def _get_extraction_ts_pths(self):
        extracted_patients = (self.df
                              .filter(pl.col('source_dataset').ne('mimic3'))
                              .select('patient')
                              .unique()
                              .collect()
                              .to_pandas()
                              .patient)
        extracted_ts_pths = self.ts_pths.loc[extracted_patients, 'ts_pth'].to_list()
        return extracted_ts_pths
    
    def get_ts_table(self, n_patients=None):
        '''
        n_patients: Number of patients to read med exposure from. by default None
        reads all patients (takes several minutes).
        '''
        pths = (self.extracted_ts_pths 
                if n_patients is None 
                else pd.Series(self.extracted_ts_pths).sample(n=n_patients).to_list())
        
        df = pl.scan_parquet(pths, low_memory=True, rechunk=False, extra_columns='ignore')
        source_datasets = self.df.select('source_dataset', 'patient')
        
        patient_counts = (df.select('patient')
                          .unique()
                          .join(source_datasets, on='patient')
                          .group_by('source_dataset')
                          .n_unique()
                          .rename({'patient': 'patient_count'}))
        
        self.kept_ts = [
            'heart_rate',
            'invasive_systolic_blood_pressure',
            'O2_arterial_saturation',
            'temperature', 
            'pH',
            'platelets'], 
        
        tab = (df
                .select(*self.kept_ts, "patient")
                .melt(id_vars='patient')
                .join(source_datasets, on='patient')
                .unique()
                .group_by("variable", "source_dataset")
                .agg([
                    pl.col('value').mean().alias('vital_mean'),
                    pl.col('value').std().alias('vital_std'),
                    ])
                .select('source_dataset','variable','vital_mean','vital_std') 
                .collect()
                .to_pandas()
                )
        
        return tab
    
    def get_unit_table(self, n_patients=None):   
        tab = (self.df
               .group_by(['source_dataset','unit_type'])
               .agg(pl.count().alias('n_unit_type'))
               .with_columns((pl.col('n_unit_type') / pl.col('n_unit_type')
                             .sum()
                             .over('source_dataset') * 100)
                             .alias('p_unit_type'))
               .filter(pl.col('source_dataset').ne('mimic3'))
               .collect()
               .to_pandas()
               )
        return tab
    
    def get_med_table(self, n_patients=None):
        pths = (self.extracted_ts_pths 
                if n_patients is None 
                else pd.Series(self.extracted_ts_pths).sample(n=n_patients).to_list())
        
        df = pl.scan_parquet(pths)
        source_datasets = self.df.select('source_dataset', 'patient')


        patient_counts = (df.select('patient')
                          .unique()
                          .join(source_datasets, on='patient')
                          .group_by('source_dataset')
                          .n_unique()
                          .rename({'patient': 'patient_count'}))

        tab = (df.select(*self.kept_meds, 'patient')
               .melt(id_vars='patient')
               .filter(pl.col('value')>0)
               .select('patient', 'variable')
               .join(source_datasets, on='patient')
               .unique()
               .group_by('variable', 'source_dataset')
               .n_unique()
               .rename({'variable': 'drug',
                        'patient': 'drug_count'})
               .join(patient_counts, on='source_dataset')
               .with_columns(
                   drug_percentage=pl.col('drug_count').truediv(pl.col('patient_count'))
                   )
               .select('source_dataset', 'drug', 'drug_percentage', 'drug_count')
               .collect()
               .to_pandas())
        Path(self.drug_exposure_savepath).parent.mkdir(exist_ok=True)
        # tab.to_parquet(self.drug_exposure_savepath)
        return tab
        
    @staticmethod
    def make_table(df_flat, df_meds):
        flats = (df_flat
                 .round(1)
                 .applymap(lambda x: "-"
                                 if x == 0 or x == 0.0
                                 else "{:,}".format(x) if isinstance(x, (int, float))
                                 else x)
                 .astype(str)
                 .assign(age_with_std=lambda x: x.age + ' ['+ x.age_std+']',
                         sex_female=lambda x: x.n_sex_female + ' ('+ x.p_sex_female+'\%)',
                         sex_male=lambda x: x.n_sex_male + ' ('+ x.p_sex_male+'\%)',
                         sex_unknown=lambda x: x.n_sex_unknown + ' ('+ x.p_sex_unknown+'\%)',
                         height_with_std=lambda x: x.height+ ' ['+ x.height_std+']',
                         weight_with_std=lambda x: x.weight+ ' ['+ x.weight_std+']',
                         los_with_std=lambda x: x.LoS+ ' ['+ x.LoS_std+']',
                         long_los=lambda x: x.n_long_los + ' ('+ x.p_long_los+'\%)',
                         mortality=lambda x: x.n_mortality + ' ('+ x.p_mortality+'\%)'
                         )
                 .drop(columns=['age',
                                'age_std',
                                'n_sex_female',
                                'p_sex_female',
                                'n_sex_male',
                                'p_sex_male',
                                'n_sex_unknown',
                                'p_sex_unknown',
                                'height',
                                'height_std',
                                'weight',
                                'weight_std',
                                'LoS',
                                'LoS_std',
                                'p_long_los',
                                'n_long_los',
                                'n_mortality',
                                'p_mortality',
                                ])
                 .set_index('source_dataset').T
                 .rename({
                     'age_with_std': r'\textbf{Age (years), mean [SD]}',
                     'sex_female': r'\indent \textbullet{} \indent \textbf{Female}', 
                     'sex_male': r'\indent \textbullet{} \indent \textbf{Male}',
                     'sex_unknown': r'\indent \textbullet{} \indent \textbf{Unknown}',
                     'height_with_std': r'\textbf{Height (cm), mean [SD]}',
                     'weight_with_std': r'\textbf{Weight (kg), mean [SD]}',
                     'los_with_std': r'\midrule \textbf{ICU LoS (hours), mean [SD]}',
                     'stays': r'\textbf{ICU Stays, n\textsuperscript{1}}',
                     'long_los': r'\textbf{ICU Los \> 10 days, n(\%)}',
                     'mortality': r'\textbf{ICU mortality, n(\%)}'
                     }))
        
        unit_types = (df_unit
                         .set_index(['source_dataset', 'unit_type'])
                         .unstack('source_dataset')
                         .fillna(0))
        unit_n_table = (unit_types["n_unit_type"]
                        .astype(int)
                        .applymap(lambda x: "-"
                                        if x == 0 or x == 0.0
                                        else "{:,}".format(x) if isinstance(x, (int, float))
                                        else x)
                        .astype(str))
        unit_p_table = (unit_types['p_unit_type']
                        .round(2)
                        .applymap(lambda x: "-"
                                        if x == 0 or x == 0.0
                                        else "{:,}".format(x) if isinstance(x, (int, float))
                                        else x)
                        .astype(str))
        unit_p_table = unit_p_table.applymap(
                                        lambda x: x if x == "-" else f' ({x}\%)'
                                        )
        formatted_unit_types = unit_n_table + unit_p_table
        formatted_unit_types.index = formatted_unit_types.index.str.capitalize()
        formatted_unit_types = formatted_unit_types.rename(index=lambda x: fr'\indent\textbullet{{}}\indent \textbf{{{x}}}')
        
        desired_order = ['heart_rate',
                         'invasive_systolic_blood_pressure',
                         'O2_arterial_saturation', 
                         'temperature',
                         'pH',
                         'platelets']
        ts = (df_ts
                 .set_index(['source_dataset', 'variable'])
                 .unstack('source_dataset')
                 .fillna(0)
                 .reindex(index = desired_order)
                 .rename({'heart_rate': 'Heart rate (/min)',
                          'invasive_systolic_blood_pressure' : 'Inv. Syst. Art. pressure (mmHg)',
                          'O2_arterial_saturation' : r'O\textsuperscript{2} Sat. in Art. blood (\%)',
                          'temperature' : 'Temperature (°C)', 
                          'pH' : 'pH', 
                          'platelets' : 'Platelets count (G/L)', 
                          }))
        
        vitals_mean_table = ts["vital_mean"].round(1).astype(str)
        vitals_std_table = ' ['+ts['vital_std'].round(1).astype(str)+']'
        
        formatted_vitals = vitals_mean_table + vitals_std_table
        formatted_vitals = formatted_vitals.rename(index=lambda x: fr'\indent\textbullet{{}}\indent \textbf{{{x}}}')
        
        meds = (df_meds
                 .set_index(['source_dataset', 'drug'])
                 .unstack('source_dataset')
                 .fillna(0))
        med_number_table = (meds['drug_count']
                            .astype(int)
                            .applymap(lambda x: "-"
                                            if x == 0 or x == 0.0
                                            else "{:,}".format(x) if isinstance(x, (int, float))
                                            else x)
                            .astype(str))
        med_percent_table = (meds['drug_percentage']
                             .mul(100)
                             .round(1)
                             .applymap(lambda x: "-"
                                             if x == 0 or x == 0.0
                                             else "{:,}".format(x) if isinstance(x, (int, float))
                                             else x)
                             .astype(str))
        # Add the % sign only to non-"-" values
        med_percent_table = med_percent_table.applymap(
                                        lambda x: x if x == "-" else f' ({x}\%)'
                                        )
        formatted_meds = med_number_table + med_percent_table
        formatted_meds.index = formatted_meds.index.str.capitalize()
        formatted_meds = formatted_meds.rename(index=lambda x: fr'\indent\textbullet{{}}\indent \textbf{{{x}}}')
       
        tab = (pd.concat([flats,
                          formatted_vitals, 
                          formatted_unit_types, 
                          formatted_meds])
               .rename_axis(None, axis=1))
        
        new_line = pd.DataFrame({
                'amsterdam': ['No'],
                'hirid': ['No'],
                'eicu': ['Yes (N=238)'],
                'mimic4': ['No']
                }).rename(index={0: r'\textbf{Multi-centric database}'})
        tab = pd.concat([tab.iloc[:0], new_line, tab.iloc[0:]])

        new_line2 = pd.DataFrame({
                'amsterdam': [''],
                'hirid': [''],
                'eicu': [''],
                'mimic4': ['']
                }).rename(index={0: r'\midrule \textbf{Demographics}'})
        tab = pd.concat([tab.iloc[:3], new_line2, tab.iloc[3:]])
       
        new_line3 = pd.DataFrame({
                'amsterdam': [''],
                'hirid': [''],
                'eicu': [''],
                'mimic4': ['']
                }).rename(index={0: r'\textbf{Sex, n (\%)}'})
        tab = pd.concat([tab.iloc[:5], new_line3, tab.iloc[5:]])
      
        new_line4 = pd.DataFrame({
                'amsterdam': [''],
                'hirid': [''],
                'eicu': [''],
                'mimic4': ['']
                }).rename(index={0: r'\midrule \textbf{Continous Variables:}'})
        tab = pd.concat([tab.iloc[:14], new_line4, tab.iloc[14:]])
        
        new_line5 = pd.DataFrame({
                'amsterdam': [''],
                'hirid': [''],
                'eicu': [''],
                'mimic4': ['']
                }).rename(index={0: r'\textbf{Vitals, mean [SD]}'})
        tab = pd.concat([tab.iloc[:15], new_line5, tab.iloc[15:]])
        
        new_line6 = pd.DataFrame({
                'amsterdam': [''],
                'hirid': [''],
                'eicu': [''],
                'mimic4': ['']
                }).rename(index={0: r'\textbf{Laboratory, mean [SD]}'})
        tab = pd.concat([tab.iloc[:20], new_line6, tab.iloc[20:]])
        
        new_line7 = pd.DataFrame({
                'amsterdam': [''],
                'hirid': [''],
                'eicu': [''],
                'mimic4': ['']
                }).rename(index={0: r'\midrule \textbf{Unit Type\textsuperscript{2}, n(\%)}'})
        tab = pd.concat([tab.iloc[:23], new_line7, tab.iloc[23:]])
        
        new_line8 = pd.DataFrame({
                'amsterdam': [''],
                'hirid': [''],
                'eicu': [''],
                'mimic4': ['']
                }).rename(index={0: r'\midrule \textbf{Drug Exposure\textsuperscript{3}, n(\%)}'})
        tab = pd.concat([tab.iloc[:33], new_line8, tab.iloc[33:]])
        
        desired_order = ['eicu', 'amsterdam', 'hirid', 'mimic4']
        tab = tab.reindex(columns = desired_order)
        tab = tab.rename(columns={'amsterdam': r'\textbf{AmsterdamUMC}',
                                'hirid': r'\textbf{HiRID} ',
                                'eicu': r'\textbf{eICU}',
                                'mimic4': r'\textbf{MIMIC-IV}',
                                })

        latext = (tab.to_latex(float_format="{:.1f}".format,
                               column_format='lccccc')
                  .replace('patient', r'\textbf{Patients}, n'))
        
        print('\n',latext)
    
    
if __name__=="__main__":
    self = ExtractionTable()
    
    df_flat = self.get_flat_table()
    df_ts = self.get_ts_table() # takes some time to run
    df_unit = self.get_unit_table()
    df_meds = self.get_med_table()
    df_meds = self.get_med_table()
    
    self.make_table(df_flat, df_meds)

    