"""Paper-assets-inspired grayscale tables: fine rules, normal font weights, no Mean."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .config import HIERARCHIES, TIME_PAIRS

TITLES = {
    'table1':'Table 1  |  Feature Extraction Ablation',
    'table2':'Table 2  |  PIJ Construction Ablation',
    'table3':'Table 3  |  Network Input Ablation',
}
SUBTITLES = {
    'table1':'GRN + CCI  /  alpha = 0.20  /  KL+OT   ·   Delta EI (bits)',
    'table2':'GRN: NMF  /  CCI: Laplacian  /  alpha = 0.20   ·   Delta EI (bits)',
    'table3':'GRN: NMF  /  CCI: Laplacian  /  KL+OT   ·   Delta EI (bits)',
}

def render_table(frame:pd.DataFrame, table_id:str, out:Path, *,dpi:int=250, preview:bool=False)->None:
    if table_id not in TITLES:raise ValueError(table_id)
    n=len(frame); cols=list(frame.columns)
    fig_h = max(5.6,2.0+n*.255)
    fig,ax=plt.subplots(figsize=(13.6,fig_h)); ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.text(.014,.990,TITLES[table_id],transform=ax.transAxes,ha='left',va='top',fontsize=15,fontweight='normal',color='#202020')
    ax.text(.014,.930,SUBTITLES[table_id],transform=ax.transAxes,ha='left',va='top',fontsize=10,fontweight='normal',color='#666666')
    if preview:
        ax.text(.986,.990,'LAYOUT PREVIEW  ·  Round 12 Pilot',transform=ax.transAxes,va='top',ha='right',fontsize=9,color='#8B8B8B')
        ax.text(.014,.884,'Only Spot/K150/K40 have pilot values; K10 cells are intentionally blank.',transform=ax.transAxes,ha='left',va='top',fontsize=8.5,color='#999999')
    body=[]
    for _,r in frame.iterrows():
        body.append([('—' if pd.isna(r[c]) else f'{float(r[c]):+.4f}') if c in TIME_PAIRS else str(r[c]) for c in cols])
    # Similar to previous Paper Assets table: hierarchy grouping, spacious columns.
    if table_id=='table1':widths=[.195,.18,.18,.148,.148,.149]
    else:widths=[.27,.205,.175,.175,.175]
    tab=ax.table(cellText=body,colLabels=cols,colWidths=widths,cellLoc='center',colLoc='center',
                 bbox=(.012,.055,.976,.780))
    tab.auto_set_font_size(False);tab.set_fontsize(9.4)
    for (r,c),cell in tab.get_celld().items():
        cell.set_facecolor('#FFFFFF');cell.set_edgecolor('#CBCBCB');cell.set_linewidth(.42)
        cell.PAD=.020
        txt=cell.get_text();txt.set_fontweight('normal');txt.set_color('#242424')
        if r==0:txt.set_color('#4A4A4A');txt.set_fontsize(9.4)
        elif c==0:txt.set_ha('left')
    # Soft hierarchy groups, no thick separators, no emphatic maxima.
    blocks = 4 if table_id=='table1' else 2 if table_id=='table2' else 3
    for k in range(len(frame["Hierarchy"].drop_duplicates())):
        first=1+k*blocks;last=first+blocks-1
        for j in range(first,last+1):
            cell=tab[(j,0)]
            if j!=first+blocks//2:cell.get_text().set_text('')
            if j==first:cell.visible_edges='TLR'
            elif j==last:cell.visible_edges='BLR'
            else:cell.visible_edges='LR'
            cell.set_edgecolor('#CBCBCB');cell.set_linewidth(.42)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=dpi,bbox_inches='tight',facecolor='white')
    fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
