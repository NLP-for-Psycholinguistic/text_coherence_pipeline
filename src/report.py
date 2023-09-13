import mdutils
import pandas as pd

def write_report(df_example,config):
    if config['use_config']:
        return write_report_wconfig(df_example,config)
    else:
        return write_report_woconfig(df_example)

    
def write_report_woconfig(df_example):
    """writes the basic markdown report with all texts, all modes and all verbatim"""
    mdFile = mdutils.MdUtils(file_name='report',title='Verbatim Results Report')
    speech_disorders_dict = {'normal' : 'None', 'max' : 'Clanging','min' : 'Derailment, Incoherence, Illogicality'}
    
    #Introduction
    mdFile.new_header(level=1, title='Introduction')
    mdFile.new_paragraph("This report is a summary of the results obtained from the extraction of the verbatims.")
    mdFile.new_paragraph(f"The extraction has highlighted {len(df_example)} verbatims, among {len(df_example['code'].unique())} different texts.")
    mdFile.new_paragraph(f"For each text, we have extracted verbatim of {len(df_example['mode'].unique())} different modes, for each verbatim, we show the score (distance to the mean similarity among all paragraphs), the scaled score (score divided by the mean of the text) and the z-score (score divided by the standard deviation of the text).")
    mdFile.new_paragraph(f"Please read the documentation for more information about the extraction process.")
    mdFile.new_line()

    #Content
    mdFile.new_header(level=1, title='Verbatim')
    for code in df_example['code'].unique():
        mdFile.new_header(level=2, title=f"Text {code}",)
        df_verbatim_code = df_example[df_example['code']==code]
        mdFile.new_paragraph(f"{len(df_verbatim_code)} verbatims were extracted from this text.")

        mdFile.new_line()
        mdFile.write("The text has a ")
        mdFile.write(f"{round(df_verbatim_code.iloc[0]['mean'],2)}",bold_italics_code='b',color='red')
        mdFile.write("mean distance to text similarity.")
        mdFile.new_line()

        for mode in df_verbatim_code['mode'].unique():
            mdFile.new_header(level=3, title=f"Verbatims of type {mode}")
            mdFile.new_paragraph(f"Possible speech disorders for these verbatim are : {speech_disorders_dict[mode]}",bold_italics_code='b')
            mdFile.new_line()
            df_verbatim_mode = df_verbatim_code[df_verbatim_code['mode']==mode]
            for index, row in df_verbatim_mode.iterrows():
                        mdFile.new_header(level=4, title=f"Verbatim {index} of mode {mode}")

                        mdFile.new_line()

                        mdFile.write("Verbatim score : ")
                        mdFile.write(f"{round(row['value'],3)}",bold_italics_code="b",color='red')

                        mdFile.write(", scaled verbatim score : ")
                        mdFile.write(f"{round(row['scaled_value'],3)}",bold_italics_code='b',color= 'red')

                        mdFile.write(" verbatim z-score : ")
                        mdFile.write(f"{round(row['z_score'],3)}",bold_italics_code='b',color= 'red')
                        mdFile.new_line()
                        
                        mdFile.new_paragraph(f"Verbatim text :")
                        mdFile.new_paragraph(f">{row['text']}")
    #TOC
    mdFile.new_table_of_contents(table_title='Table of content', depth=3)
    return mdFile

def write_report_wconfig(df_example,config):
    """Applies config option to the markdown report"""
    if config['hidden_score']:
        if config['mode_selection']!= 'all':
            df_example = df_example[df_example['mode'].isin(config['mode_selection'])]

        if config['sample']!=1:
            df_example = df_example.sample(frac=config['sample'])

        return write_report_hidden(df_example)
    elif config['sort_by_score'] or config['sort_by_external_column']:
        sort_column = ('scaled_value' if config['sort_by_score'] == False else config['sort_by_external_column'])
        assert sort_column in df_example.columns

        if config['mode_selection'] != 'all':
            df_example = df_example[df_example['mode'].isin(config['mode_selection'])]

        if config['sample']!=1:
            df_example = df_example.head(int(config['sample']*len(df_example)))

        df_example[sort_column + 'abs'] = df_example[sort_column].abs()

        df_example = df_example.sort_values(by=sort_column + 'abs',ascending=False)
        return write_report_by_order(df_example)
    elif config['mode_selection'] != 'all':
        df_example = df_example[df_example['mode'].isin(config['mode_selection'])]
        if config['sample']!=1:
            df_example = df_example.sample(frac=config['sample'])
        return write_report_woconfig(df_example)
    else:
        return write_report_woconfig(df_example)

def write_report_hidden(df_example):
    """writes a markdown report with the values of the verbatim hidden and reported in a table at the end, for manual labeling"""
    mdFile = mdutils.MdUtils(file_name='report',title='Verbatim Results Report')
    speech_disorders_dict = {'normal' : 'None', 'max' : 'Clanging','min' : 'Derailment, Incoherence, Illogicality'}
    
    #Introduction
    mdFile.new_header(level=1, title='Introduction')
    mdFile.new_paragraph("This report is a summary of the results obtained from the extraction of the verbatims.")
    mdFile.new_paragraph(f"The extraction has highlighted {len(df_example)} verbatims, among {len(df_example['code'].unique())} different texts.")
    mdFile.new_paragraph(f"For each verbatim, we show the score (distance to the mean similarity among all paragraphs), the scaled score (score divided by the mean of the text) and the z-score (score divided by the standard deviation of the text) in a table at the end of the report.")
    mdFile.new_paragraph(f"Please read the documentation for more information about the extraction process.")
    mdFile.new_line()

    #Content
    mdFile.new_header(level=1, title='Verbatim')
    mdFile.new_line()
    for index, row in df_example.iterrows():
                mdFile.new_header(level=2, title=f"Verbatim {index}")
                mdFile.new_line()
                mdFile.write(mdFile.new_reference_link(link="#score-table",text=f"score table"))
                mdFile.new_paragraph(f"Verbatim text :")
                mdFile.new_paragraph(f">{row['text']}")

    #Score table
    mdFile.new_line()
    mdFile.new_header(level=1, title='Score table')
    table_content = ['Verbatim number', 'Verbatim score', 'Scaled verbatim score', 'Verbatim z-score','possible speech disorder']
    df_example = df_example.sort_index()
    for index, row in df_example.iterrows():
        table_content.extend([f"{index}",f"{round(row['scaled_value'],3)}",f"{round(row['scaled_value'],3)}",f"{round(row['z_score'],3)}",f"{speech_disorders_dict[row['mode']]}"])

    mdFile.new_table(columns=5,rows=len(df_example)+1,text=table_content)
    #TOC
    mdFile.new_table_of_contents(table_title='Table of content', depth=3)
    return mdFile

def write_report_by_order(df_example):
    """Writes a markdown report with the verbatim in the order of the dataframe"""

    mdFile = mdutils.MdUtils(file_name='report',title='Verbatim Results Report')
    speech_disorders_dict = {'normal' : 'None', 'max' : 'Clanging','min' : 'Derailment, Incoherence, Illogicality'}
    
    #Introduction
    mdFile.new_header(level=1, title='Introduction')
    mdFile.new_paragraph("This report is a summary of the results obtained from the extraction of the verbatims.")
    mdFile.new_paragraph(f"The extraction has highlighted {len(df_example)} verbatims, among {len(df_example['code'].unique())} different texts.")
    mdFile.new_paragraph(f"For each text, we have extracted verbatim of {len(df_example['mode'].unique())} different modes, for each verbatim, we show the score (distance to the mean similarity among all paragraphs), the scaled score (score divided by the mean of the text) and the z-score (score divided by the standard deviation of the text).")
    mdFile.new_paragraph("Verbatims are sorted by a column defined in the config file, either the scaled value or an external column.")
    mdFile.new_paragraph(f"Please read the documentation for more information about the extraction process.")
    mdFile.new_line()

    #Content
    mdFile.new_header(level=1, title='Verbatim')
    mdFile.new_line()
    for index, row in df_example.iterrows():
                mode = row['mode']
                mdFile.new_paragraph(f"Possible speech disorders for these verbatim are : {speech_disorders_dict[mode]}",bold_italics_code='b')
                mdFile.new_header(level=2, title=f"Verbatim {index} of mode {mode}")

                mdFile.new_line()

                mdFile.write("Verbatim score : ")
                mdFile.write(f"{round(row['value'],3)}",bold_italics_code="b",color='red')

                mdFile.write(", scaled verbatim score : ")
                mdFile.write(f"{round(row['scaled_value'],3)}",bold_italics_code='b',color= 'red')

                mdFile.write(" verbatim z-score : ")
                mdFile.write(f"{round(row['z_score'],3)}",bold_italics_code='b',color= 'red')
                mdFile.new_line()
                
                mdFile.new_paragraph(f"Verbatim text :")
                mdFile.new_paragraph(f">{row['text']}")
    #TOC
    mdFile.new_table_of_contents(table_title='Table of content', depth=3)
    return mdFile