from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from mignet_ce.visualization.downstream.style import (
    BLUE, CYAN, DARK, GOLD, GREEN, GRID, LIGHT, MUTED, NAVY, PURPLE, RED,
    DIVERGING, SEQUENTIAL, add_panel_label, savefig, set_publication_style,
)

MAPPINGS=("Seurat K150","Seurat K40","Optimized coarse-graining")
COLORS={"Seurat K150":CYAN,"Seurat K40":BLUE,"Optimized coarse-graining":RED,"Spot (uncompressed)":MUTED}
MARKERS={"Seurat K150":"o","Seurat K40":"s","Optimized coarse-graining":"^","Spot (uncompressed)":"D"}
PAIRS=("11.5->12.5","12.5->13.5","13.5->14.5")
WINDOWS=("11.5->12.5->13.5","12.5->13.5->14.5")


def _panel(ax,label,title): add_panel_label(ax,label); ax.set_title(title,pad=7)
def _save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True); path=out/f"{name}.png";savefig(fig,path);return path


def plot_markov_memory(summary:pd.DataFrame, states:pd.DataFrame, corr:pd.DataFrame, out:Path)->Path:
    set_publication_style(); fig,axes=plt.subplots(2,3,figsize=(14.2,8.2),constrained_layout=True)
    x=np.arange(len(WINDOWS)); width=.24
    metrics=[("markov_memory_cmi_bits","History adds predictive information","I(past; future | current) (bit)"),
             ("history_gain_fraction_of_uncertainty","Fraction of uncertainty explained by history","History gain / H(future | current)"),
             ("memory_to_long_mi_ratio","Memory relative to long-range signal","CMI / I(past; future)")]
    for idx,(metric,title,ylabel) in enumerate(metrics):
        ax=axes[0,idx]
        for mi,m in enumerate(MAPPINGS):
            s=summary[summary.mapping==m].set_index('window').reindex(WINDOWS)
            ax.bar(x+(mi-1)*width,s[metric],width*.9,color=COLORS[m],label=m)
        ax.set_xticks(x,[w.replace('->','–') for w in WINDOWS],rotation=17,ha='right');ax.set_ylabel(ylabel);ax.grid(axis='y');_panel(ax,chr(65+idx),title)
        if idx==0:ax.legend(fontsize=6.8)
    ax=axes[1,0]
    for m in MAPPINGS:
        s=states[states.mapping==m]; ax.scatter(s['state_probability'],s['state_memory_bits'],s=14,alpha=.42,color=COLORS[m],label=m)
    ax.set_xlabel('Intermediate-state probability');ax.set_ylabel('State-specific memory (bit)');ax.grid(True);_panel(ax,'D','Where macro memory is concentrated')
    ax=axes[1,1]
    for m in MAPPINGS:
        s=states[(states.mapping==m)&states['next_step_lumpability_mean_js'].notna()]; ax.scatter(s['next_step_lumpability_mean_js'],s['state_memory_bits'],s=14,alpha=.42,color=COLORS[m],label=m)
    ax.set_xlabel('Next-step intrinsic closure residual (mean JS)');ax.set_ylabel('State-specific memory (bit)');ax.grid(True);_panel(ax,'E','Memory versus approximate lumpability')
    ax=axes[1,2]
    # Show Spearman correlations with intrinsic residual and Q mismatch.
    labels=[];vals=[];cols=[]
    mapping_labels={"Seurat K150":"K150","Seurat K40":"K40","Optimized coarse-graining":"Optimized"}
    for m in MAPPINGS:
        for metric,short in [('next_step_lumpability_mean_js','Intrinsic'),('next_step_q_best_direct_js','Q mismatch')]:
            g=corr[(corr.mapping==m)&(corr.x==metric)]
            labels.append(f"{mapping_labels[m]}\n{short}"); vals.append(float(g['spearman_rho'].mean()) if len(g) else np.nan); cols.append(COLORS[m])
    ax.bar(np.arange(len(vals)),vals,color=cols);ax.axhline(0,color=MUTED,ls='--',lw=1);ax.set_xticks(np.arange(len(vals)),labels,rotation=24,ha='right',fontsize=6.2);ax.set_ylabel('Mean Spearman rho');ax.grid(axis='y');_panel(ax,'F','Does memory track closure failure?')
    return _save(fig,out,'closure_markov_memory_order')


def plot_predictive_frontier(frame:pd.DataFrame,out:Path)->Path:
    set_publication_style(); fig,axes=plt.subplots(2,3,figsize=(14.2,8.1),constrained_layout=True)
    ax=axes[0,0]
    for m in (*MAPPINGS,'Spot (uncompressed)'):
        s=frame[frame.mapping==m]; ax.scatter(s['macro_entropy_bits'],s['predictive_information_bits'],s=75,marker=MARKERS[m],color=COLORS[m],label=m)
        for _,r in s.iterrows(): ax.annotate(r['time_pair'].split('->')[0],(r['macro_entropy_bits'],r['predictive_information_bits']),xytext=(4,3),textcoords='offset points',fontsize=6)
    ax.set_xlabel('Macro representation entropy H(A_t) (bit)');ax.set_ylabel('Retained predictive information (bit)');ax.grid(True);ax.legend(fontsize=6.4);_panel(ax,'A','Predictive compression frontier')
    ax=axes[0,1]
    for m in MAPPINGS:
        s=frame[frame.mapping==m]; ax.scatter(s['compression_fraction'],s['macro_sufficiency'],s=70,marker=MARKERS[m],color=COLORS[m],label=m)
    ax.set_xlabel('Representation entropy / spot entropy');ax.set_ylabel('Macro sufficiency');ax.set_xlim(0,1.05);ax.set_ylim(0,1.05);ax.grid(True);_panel(ax,'B','Compression versus dynamical sufficiency')
    ax=axes[0,2]
    for m in MAPPINGS:
        s=frame[frame.mapping==m]; ax.scatter(s['closure_floor_js'],s['predictive_efficiency_bits_per_state_bit'],s=70,marker=MARKERS[m],color=COLORS[m],label=m)
    ax.set_xlabel('Intrinsic closure floor (mean JS)');ax.set_ylabel('Predictive bits / representation bit');ax.grid(True);_panel(ax,'C','Efficiency–closure trade-off')
    x=np.arange(len(PAIRS));width=.24
    for idx,(metric,title,ylabel) in enumerate([('effective_states','Effective state count','K_eff'),('macro_sufficiency','Retained future information','Sufficiency'),('independent_q_excess_js','Independent-Q penalty','Excess JS')]):
        ax=axes.ravel()[3+idx]
        for mi,m in enumerate(MAPPINGS):
            s=frame[frame.mapping==m].set_index('time_pair').reindex(PAIRS);ax.bar(x+(mi-1)*width,s[metric],width*.9,color=COLORS[m],label=m)
        ax.set_xticks(x,[p.replace('->','–') for p in PAIRS],rotation=20,ha='right');ax.set_ylabel(ylabel);ax.grid(axis='y');_panel(ax,chr(68+idx),title)
    return _save(fig,out,'closure_predictive_compression_frontier')


def plot_failure_taxonomy(states:pd.DataFrame,summary:pd.DataFrame,out:Path)->Path:
    set_publication_style(); fig,axes=plt.subplots(2,3,figsize=(14.4,8.2),constrained_layout=True)
    classes=['Typical/closed','Partition-limited','Q-limited','Dual-limited']; class_colors={'Typical/closed':GREEN,'Partition-limited':GOLD,'Q-limited':PURPLE,'Dual-limited':RED}
    for c,pair in enumerate(PAIRS):
        ax=axes[0,c]
        for m in MAPPINGS:
            s=states[(states.mapping==m)&(states.time_pair==pair)]
            ax.scatter(s['mean_js'],s['q_best_direct_js'],s=np.clip(s['mass'],8,80),alpha=.42,color=COLORS[m],label=m)
        ax.set_xlabel('Intrinsic partition residual (mean JS)');ax.set_ylabel('Induced-vs-independent Q mismatch (JS)');ax.grid(True);_panel(ax,chr(65+c),f'Failure anatomy | {pair.replace("->","–")}')
        if c==0:ax.legend(fontsize=6.5)
    ax=axes[1,0]
    # Stacked fraction averaged across time pairs.
    avg=summary.groupby(['mapping','failure_class'])['fraction'].mean().unstack(fill_value=0).reindex(index=MAPPINGS,columns=classes,fill_value=0)
    bottom=np.zeros(len(MAPPINGS))
    for cls in classes:
        vals=avg[cls].to_numpy(float);ax.bar(np.arange(len(MAPPINGS)),vals,bottom=bottom,color=class_colors[cls],label=cls);bottom+=vals
    ax.set_xticks(np.arange(len(MAPPINGS)),['K150','K40','Optimized']);ax.set_ylabel('Fraction of states');ax.set_ylim(0,1);ax.legend(fontsize=6.3);ax.grid(axis='y');_panel(ax,'D','High-tail failure taxonomy')
    ax=axes[1,1]
    share=summary.groupby(['mapping','failure_class'])['residual_kl_share'].mean().unstack(fill_value=0).reindex(index=MAPPINGS,columns=classes,fill_value=0)
    bottom=np.zeros(len(MAPPINGS))
    for cls in classes:
        vals=share[cls].to_numpy(float);ax.bar(np.arange(len(MAPPINGS)),vals,bottom=bottom,color=class_colors[cls],label=cls);bottom+=vals
    ax.set_xticks(np.arange(len(MAPPINGS)),['K150','K40','Optimized']);ax.set_ylabel('Share of intrinsic residual KL');ax.grid(axis='y');_panel(ax,'E','Which state class carries the error?')
    ax=axes[1,2]
    for m in MAPPINGS:
        s=states[states.mapping==m]; ax.scatter(s['mass'],s['mean_js']+s['q_best_direct_js'],s=14,alpha=.4,color=COLORS[m],label=m)
    ax.set_xscale('log');ax.set_xlabel('State mass (log scale)');ax.set_ylabel('Intrinsic + Q mismatch');ax.grid(True);_panel(ax,'F','Are small states disproportionately problematic?')
    return _save(fig,out,'closure_failure_taxonomy')


def plot_transition_mismatch(summary:pd.DataFrame,edges:pd.DataFrame,out:Path)->Path:
    set_publication_style();fig,axes=plt.subplots(2,3,figsize=(14.2,8.1),constrained_layout=True);x=np.arange(len(PAIRS));width=.24
    metrics=[('weighted_l1_mismatch','Total weighted transition mismatch','Weighted L1'),('top25_edge_share','Top-25 transitions explain','Fraction of total mismatch'),('edge_mismatch_gini','Mismatch concentration','Gini coefficient')]
    for idx,(metric,title,ylabel) in enumerate(metrics):
        ax=axes[0,idx]
        for mi,m in enumerate(MAPPINGS):
            s=summary[summary.mapping==m].set_index('time_pair').reindex(PAIRS);ax.bar(x+(mi-1)*width,s[metric],width*.9,color=COLORS[m],label=m)
        ax.set_xticks(x,[p.replace('->','–') for p in PAIRS],rotation=20,ha='right');ax.set_ylabel(ylabel);ax.grid(axis='y');_panel(ax,chr(65+idx),title)
        if idx==0:ax.legend(fontsize=6.6)
    # Bottom row: ranked top edge mismatches per mapping, pooled across time.
    for idx,m in enumerate(MAPPINGS):
        ax=axes[1,idx]; s=edges[edges.mapping==m].copy(); s=s.sort_values('share_of_total_abs_mismatch',ascending=False).head(15).reset_index(drop=True)
        signs=np.where(s['signed_weighted_difference']>=0,RED,BLUE)
        ax.barh(np.arange(len(s)),s['share_of_total_abs_mismatch'],color=signs);ax.invert_yaxis();
        labels=[f"{r.time_pair.split('->')[0]}: {int(r.source_state)}→{int(r.target_state)}" for _,r in s.iterrows()]
        ax.set_yticks(np.arange(len(s)),labels,fontsize=6);ax.set_xlabel('Share of absolute mismatch');ax.grid(axis='x');_panel(ax,chr(68+idx),f'Top mismatched transitions | {m.replace("Seurat ","")}')
    return _save(fig,out,'closure_transition_mismatch_anatomy')


def plot_subspace_alignment(frame:pd.DataFrame,out:Path)->Path:
    set_publication_style();fig,axes=plt.subplots(2,3,figsize=(14.2,8.1),constrained_layout=True)
    ranks=sorted(frame['rank'].unique())
    for idx,(metric,title,ylabel) in enumerate([('left_subspace_cosine_mean','Source dynamical subspace alignment','Mean cos(principal angle)'),('right_subspace_cosine_mean','Target dynamical subspace alignment','Mean cos(principal angle)'),('singular_value_cosine','Spectral-shape similarity','Cosine similarity')]):
        ax=axes[0,idx]
        for m in MAPPINGS:
            s=frame[(frame.mapping==m)&(frame.time_pair=='11.5->12.5')].sort_values('rank');ax.plot(s['rank'],s[metric],marker=MARKERS[m],color=COLORS[m],label=m)
        ax.set_xticks(ranks);ax.set_ylim(0,1.03);ax.set_xlabel('Top-r singular subspace');ax.set_ylabel(ylabel);ax.grid(True);_panel(ax,chr(65+idx),title+' | 11.5–12.5')
        if idx==0:ax.legend(fontsize=6.6)
    # rank-5 across time pairs
    r5=frame[frame['rank']==5]
    for idx,(metric,title) in enumerate([('left_subspace_cosine_mean','Source modes across developmental time'),('right_subspace_cosine_mean','Target modes across developmental time'),('dominant_right_vector_alignment','Dominant target mode alignment')]):
        ax=axes[1,idx];x=np.arange(len(PAIRS))
        for m in MAPPINGS:
            s=r5[r5.mapping==m].set_index('time_pair').reindex(PAIRS);ax.plot(x,s[metric],marker=MARKERS[m],color=COLORS[m],label=m)
        ax.set_xticks(x,[p.replace('->','–') for p in PAIRS],rotation=20,ha='right');ax.set_ylim(0,1.03);ax.set_ylabel('Alignment');ax.grid(True);_panel(ax,chr(68+idx),title)
    return _save(fig,out,'closure_dynamic_subspace_alignment')


def plot_information_anatomy(frame:pd.DataFrame,out:Path)->Path:
    set_publication_style();fig,axes=plt.subplots(2,3,figsize=(14.2,8.0),constrained_layout=True);x=np.arange(len(PAIRS));width=.24
    metrics=[('upward_macro_sufficiency','Upward predictive sufficiency','Fraction retained'),('upward_residual_fraction','Residual micro detail after macro','Fraction residual'),('downward_future_micro_reach','Downward reach to future micro','Fraction retained'),('macro_future_macro_bits','Macro → future macro information','Information (bit)'),('macro_future_micro_bits','Macro → future micro information','Information (bit)')]
    for idx,(metric,title,ylabel) in enumerate(metrics):
        ax=axes.ravel()[idx]
        for mi,m in enumerate(MAPPINGS):
            s=frame[frame.mapping==m].set_index('time_pair').reindex(PAIRS);ax.bar(x+(mi-1)*width,s[metric],width*.9,color=COLORS[m],label=m)
        ax.set_xticks(x,[p.replace('->','–') for p in PAIRS],rotation=20,ha='right');ax.set_ylabel(ylabel);ax.grid(axis='y');_panel(ax,chr(65+idx),title)
        if idx==0:ax.legend(fontsize=6.5)
    ax=axes.ravel()[5]
    for m in MAPPINGS:
        s=frame[frame.mapping==m];ax.scatter(s['closure_floor_js'],s['upward_macro_sufficiency'],s=72,marker=MARKERS[m],color=COLORS[m],label=m)
        for _,r in s.iterrows():ax.annotate(r['time_pair'].split('->')[0],(r['closure_floor_js'],r['upward_macro_sufficiency']),xytext=(4,3),textcoords='offset points',fontsize=6)
    ax.set_xlabel('Intrinsic closure floor (mean JS)');ax.set_ylabel('Upward macro sufficiency');ax.grid(True);_panel(ax,'F','Closure–sufficiency landscape')
    return _save(fig,out,'closure_cross_scale_information_anatomy')


def render_ultradeep_figures(*,memory_summary,memory_states,memory_corr,predictive,failure_states,failure_summary,mismatch_summary,mismatch_edges,subspace,info_anatomy,repair_curve,repair_splits,output_dir:Path):
    return [plot_markov_memory(memory_summary,memory_states,memory_corr,output_dir),plot_predictive_frontier(predictive,output_dir),plot_failure_taxonomy(failure_states,failure_summary,output_dir),plot_transition_mismatch(mismatch_summary,mismatch_edges,output_dir),plot_subspace_alignment(subspace,output_dir),plot_information_anatomy(info_anatomy,output_dir),plot_closure_repairability(repair_curve,repair_splits,output_dir)]


def plot_closure_repairability(curve:pd.DataFrame,splits:pd.DataFrame,out:Path)->Path:
    set_publication_style();fig,axes=plt.subplots(2,3,figsize=(14.2,8.1),constrained_layout=True)
    # A-C: normalized intrinsic floor under targeted state refinement.
    for idx,pair in enumerate(PAIRS):
        ax=axes[0,idx]
        for m in MAPPINGS:
            s=curve[(curve.mapping==m)&(curve.time_pair==pair)].sort_values('added_states')
            ax.plot(s['added_states'],s['fraction_floor_remaining'],marker=MARKERS[m],color=COLORS[m],label=m)
        ax.axhline(1,color=MUTED,lw=1,ls='--');ax.set_ylim(0,1.05);ax.set_xlabel('Targeted state splits added');ax.set_ylabel('Intrinsic floor / baseline');ax.grid(True);_panel(ax,chr(65+idx),f'Closure repair curve | {pair.replace("->","–")}')
        if idx==0:ax.legend(fontsize=6.5)
    # D: five-split recovery (or final available step if fewer).
    ax=axes[1,0];x=np.arange(len(PAIRS));width=.24
    for mi,m in enumerate(MAPPINGS):
        vals=[]
        for pair in PAIRS:
            s=curve[(curve.mapping==m)&(curve.time_pair==pair)].sort_values('added_states')
            r=s[s.added_states<=5].tail(1);vals.append(float(r['fraction_floor_removed'].iloc[0]) if len(r) else np.nan)
        ax.bar(x+(mi-1)*width,vals,width*.9,color=COLORS[m],label=m)
    ax.set_xticks(x,[p.replace('->','–') for p in PAIRS],rotation=20,ha='right');ax.set_ylabel('Fraction of intrinsic floor removed');ax.set_ylim(0,1);ax.grid(axis='y');_panel(ax,'D','How much do five targeted splits repair?')
    # E: targeted-vs-random matched split gain.
    ax=axes[1,1]
    offset=0
    for m in MAPPINGS:
        s=splits[splits.mapping==m];ratio=s['gain_over_random_ratio'].replace([np.inf,-np.inf],np.nan)
        xx=np.arange(len(ratio))+offset;ax.scatter(xx,ratio,s=24,alpha=.68,color=COLORS[m],label=m);offset += len(ratio)+2
    ax.axhline(1,color=MUTED,ls='--',lw=1);ax.set_yscale('log');ax.set_xlabel('Targeted split events (grouped by scale)');ax.set_ylabel('Gain / matched-random gain (log)');ax.grid(True,which='both');_panel(ax,'E','Is the hidden substructure non-random?')
    # F: size is not the whole story.
    ax=axes[1,2]
    for m in MAPPINGS:
        s=splits[splits.mapping==m];ax.scatter(s['state_mass'],s['targeted_gain_js'],s=28,alpha=.6,color=COLORS[m],label=m)
    ax.set_xscale('log');ax.set_xlabel('Split state mass (log scale)');ax.set_ylabel('Closure-floor reduction (JS)');ax.grid(True);_panel(ax,'F','Which states are most repairable?')
    return _save(fig,out,'closure_state_refinement_repairability')
