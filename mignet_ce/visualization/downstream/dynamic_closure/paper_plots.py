from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ..style import NAVY,BLUE,RED,CYAN,GOLD,GREEN,PURPLE,MUTED,SEQUENTIAL,set_publication_style,savefig,add_panel_label

MAPPINGS=("Seurat K150","Seurat K40","Optimized coarse-graining")
ALL=("Spot (uncompressed)",)+MAPPINGS
COLORS={"Spot (uncompressed)":MUTED,"Seurat K150":BLUE,"Seurat K40":CYAN,"Optimized coarse-graining":RED}
MARKERS={"Spot (uncompressed)":"o","Seurat K150":"s","Seurat K40":"^","Optimized coarse-graining":"D"}
PAIRS=("11.5->12.5","12.5->13.5","13.5->14.5")


def _save(fig,out,name):
    p=Path(out)/f"{name}.png";savefig(fig,p);return p

def _panel(ax,label,title):add_panel_label(ax,label);ax.set_title(title,loc='left')

def _pair_labels(vals):return [str(v).replace('->','–') for v in vals]


def plot_core_audit(full,lump_summary,out):
    set_publication_style();fig,axes=plt.subplots(1,3,figsize=(14.5,4.8),constrained_layout=True)
    # A: information budget, three mappings x three pairs
    ax=axes[0]; rows=[]
    for m in MAPPINGS:
        for p in PAIRS:
            r=full[(full.mapping==m)&(full.time_pair==p)].iloc[0];rows.append((m,p,float(r.I_macro_to_future_macro),float(r.closure_leakage_bits),str(r.signal_status)))
    x=np.arange(len(rows));ret=np.array([r[2] for r in rows]);leak=np.array([r[3] for r in rows])
    ax.bar(x,ret,color=BLUE,label='Macro retained information')
    ax.bar(x,leak,bottom=ret,color=RED,alpha=.78,label='Micro leakage after macro')
    for i,r in enumerate(rows):
        if r[4]=='low-signal': ax.scatter(i,ret[i]+leak[i],s=55,facecolors='white',edgecolors=MUTED,zorder=5)
    ax.set_xticks(x,[f"{m.replace('Seurat ','').replace('Optimized coarse-graining','Opt.')}\n{p.split('->')[0]}" for m,p,*_ in rows],rotation=52,ha='right')
    ax.set_ylabel('Information (bit)');ax.grid(axis='y');ax.legend(fontsize=7);_panel(ax,'A','Closure information budget')
    # B lumpability tail
    ax=axes[1];xx=np.arange(len(PAIRS));
    for m in MAPPINGS:
        s=lump_summary[lump_summary.mapping==m].set_index('time_pair').reindex(PAIRS)
        ax.plot(xx,s.mean_js,marker=MARKERS[m],color=COLORS[m],label=m)
        ax.plot(xx,s.p95_js,marker=MARKERS[m],color=COLORS[m],ls='--',alpha=.75)
    ax.set_xticks(xx,_pair_labels(PAIRS),rotation=20,ha='right');ax.set_ylabel('JS divergence');ax.grid(True)
    handles=[]
    for m in MAPPINGS:handles.append(Line2D([0],[0],marker=MARKERS[m],color=COLORS[m],label=m))
    handles+= [Line2D([0],[0],color='black',label='State-balanced mean'),Line2D([0],[0],color='black',ls='--',label='95th percentile')]
    ax.legend(handles=handles,fontsize=6.6);_panel(ax,'B','Approximate lumpability: mean and tail')
    # C independent vs induced operational gap
    ax=axes[2];xx=np.arange(len(PAIRS));width=.24
    for mi,m in enumerate(MAPPINGS):
        s=full[full.mapping==m].set_index('time_pair').reindex(PAIRS)
        induced=s.induced_closure_mean_js.to_numpy(float); direct=s.direct_closure_mean_js.to_numpy(float)
        ax.bar(xx+(mi-1)*width,induced,width*.85,color=COLORS[m],alpha=.35)
        ax.scatter(xx+(mi-1)*width,direct,s=40,marker=MARKERS[m],color=COLORS[m])
        for j in range(len(xx)):ax.vlines(xx[j]+(mi-1)*width,induced[j],direct[j],color=COLORS[m],lw=1.2)
    ax.set_xticks(xx,_pair_labels(PAIRS),rotation=20,ha='right');ax.set_ylabel('State-balanced mean JS');ax.grid(axis='y')
    ax.legend(handles=[Line2D([0],[0],marker='s',ls='',markerfacecolor='lightgray',markeredgecolor='none',label='Induced/oracle Q'),Line2D([0],[0],marker='o',ls='',color=NAVY,label='Separately estimated Q')],fontsize=7)
    _panel(ax,'C','Operational macro-Q gap')
    return _save(fig,out,'closure_core_dynamical_audit')


def plot_pareto(pareto,out):
    set_publication_style();fig,axes=plt.subplots(1,3,figsize=(14.5,4.8),constrained_layout=True)
    # common normalization for predictive color
    vals=pareto.macro_predictive_bits.to_numpy(float);vmin=np.nanmin(vals);vmax=np.nanmax(vals);norm=plt.Normalize(vmin,vmax)
    ax=axes[0]
    for _,r in pareto.iterrows():
        face=SEQUENTIAL(norm(r.macro_predictive_bits)) if r.signal_status=='informative' else 'white'
        ax.scatter(r.delta_ei,r.closure_leakage_bits,s=35+20*np.log2(max(r.active_states,2)),marker=MARKERS[r.mapping],facecolor=face,edgecolor=COLORS[r.mapping],linewidth=1.2,alpha=.95)
        ax.annotate(f"{r.mapping.replace('Seurat ','').replace('Optimized coarse-graining','Opt.')} {r.time_pair.split('->')[0]}",(r.delta_ei,r.closure_leakage_bits),xytext=(4,3),textcoords='offset points',fontsize=5.7)
    ax.axvline(0,color=MUTED,ls='--',lw=.8);ax.set_xlabel('ΔEI (bit; higher is better)');ax.set_ylabel('Closure leakage (bit; lower is better)');ax.grid(True);_panel(ax,'A','Causal emergence vs dynamical closure')
    sm=plt.cm.ScalarMappable(norm=norm,cmap=SEQUENTIAL);cb=fig.colorbar(sm,ax=ax,fraction=.046,pad=.03);cb.set_label('Macro predictive information (bit)',fontsize=7)
    ax=axes[1]
    for _,r in pareto.iterrows():
        fc=RED if r.closure_leakage_bits>np.nanmedian(pareto.closure_leakage_bits) else BLUE
        ax.scatter(r.compression_fraction,r.macro_predictive_bits,s=35+20*np.log2(max(r.active_states,2)),marker=MARKERS[r.mapping],facecolor=fc,edgecolor=COLORS[r.mapping],alpha=.7)
        ax.annotate(f"{r.mapping.replace('Seurat ','').replace('Optimized coarse-graining','Opt.')} {r.time_pair.split('->')[0]}",(r.compression_fraction,r.macro_predictive_bits),xytext=(4,3),textcoords='offset points',fontsize=5.7)
    ax.set_xlabel('Macro capacity / micro capacity (lower = stronger compression)');ax.set_ylabel('Macro predictive information (bit)');ax.grid(True);_panel(ax,'B','Compression vs non-trivial predictive content')
    ax=axes[2]
    for _,r in pareto[pareto.mapping!='Spot (uncompressed)'].iterrows():
        alpha=1.0 if not r.pareto_dominated else .35
        ax.scatter(r.crossfit_closure_kl_bits,r.independent_q_excess_js,s=55,marker=MARKERS[r.mapping],color=COLORS[r.mapping],alpha=alpha)
        ax.annotate(f"{r.mapping.replace('Seurat ','').replace('Optimized coarse-graining','Opt.')} {r.time_pair.split('->')[0]}",(r.crossfit_closure_kl_bits,r.independent_q_excess_js),xytext=(4,3),textcoords='offset points',fontsize=5.7)
    ax.set_xlabel('Cross-fitted intrinsic closure error (KL bit)');ax.set_ylabel('Separately estimated-Q excess JS');ax.grid(True);_panel(ax,'C','Partition autonomy vs separately estimated macro dynamics')
    fig.text(.5,.002,'Pareto positioning uses four anchors per time pair; no tuned frontier is claimed without a K/seed candidate sweep.',ha='center',fontsize=7,color=MUTED)
    return _save(fig,out,'causal_emergence_closure_pareto_positioning')


def plot_horizon(horizon,out):
    set_publication_style();fig,axes=plt.subplots(1,3,figsize=(14.5,4.7),constrained_layout=True)
    order=('11.5->12.5','11.5->13.5','11.5->14.5','12.5->13.5','12.5->14.5','13.5->14.5')
    for i,m in enumerate(MAPPINGS):
        ax=axes[i];s=horizon[horizon.mapping==m].set_index('interval').reindex(order).dropna(how='all').reset_index();x=np.arange(len(s))
        ax.bar(x,s.macro_information,color=BLUE,label='Macro retained')
        ax.bar(x,s.residual_information,bottom=s.macro_information,color=RED,alpha=.75,label='Micro leakage')
        ax.set_xticks(x,_pair_labels(s['interval']),rotation=32,ha='right');ax.set_ylabel('Endpoint information (bit)');ax.grid(axis='y')
        ax2=ax.twinx();valid=s.semigroup_excess_js.notna();ax2.plot(x[valid],s.loc[valid,'semigroup_excess_js'],marker='D',color=NAVY,lw=1.3,label='CK composition excess JS');ax2.set_ylabel('CK composition excess JS',color=NAVY);ax2.tick_params(axis='y',labelcolor=NAVY)
        _panel(ax,chr(65+i),f'Horizon information budget | {m.replace("Seurat ","")}')
        if i==0:ax.legend(loc='upper right',fontsize=6.8)
    return _save(fig,out,'closure_horizon_information_and_semigroup')


def plot_state_failure(source,lump_states,out):
    set_publication_style();fig,axes=plt.subplots(2,3,figsize=(14.5,8.0),constrained_layout=True);pair='11.5->12.5'
    for i,m in enumerate(MAPPINGS):
        ax=axes[0,i];s=source[(source.mapping==m)&(source.time_pair==pair)]
        sc=ax.scatter(s.x,s.y,c=s.closure_js,s=9,cmap=SEQUENTIAL,edgecolors='none');ax.set_aspect('equal');ax.invert_yaxis();ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(sc,ax=ax,fraction=.045,pad=.02,label='Closure JS')
        _panel(ax,chr(65+i),f'Spatial closure residual | {m.replace("Seurat ","")}')
    for i,m in enumerate(MAPPINGS):
        ax=axes[1,i];s=lump_states[lump_states.mapping==m]
        groups=[]
        for p in PAIRS:
            g=s[s.time_pair==p];groups.append((g.mean_js.mean(),g.p95_js.mean(),g.residual_kl_share.sort_values(ascending=False).head(min(10,len(g))).sum()))
        x=np.arange(len(PAIRS));mean=[g[0] for g in groups];p95=[g[1] for g in groups];share=[g[2] for g in groups]
        ax.plot(x,mean,marker='o',color=BLUE,label='Mean state JS');ax.plot(x,p95,marker='s',ls='--',color=RED,label='Mean state p95 JS');ax.set_xticks(x,_pair_labels(PAIRS),rotation=20,ha='right');ax.set_ylabel('State-level closure error');ax.grid(True)
        ax2=ax.twinx();ax2.plot(x,share,marker='D',color=NAVY,alpha=.65,label='Top-10 residual share');ax2.set_ylabel('Top-10 share',color=NAVY);ax2.tick_params(axis='y',labelcolor=NAVY)
        _panel(ax,chr(68+i),f'Failure concentration | {m.replace("Seurat ","")}')
        if i==0:ax.legend(fontsize=6.7)
    return _save(fig,out,'closure_state_failure_and_spatial_localization')


def plot_robustness(weighting,bootstrap,soft_hard,out):
    set_publication_style();fig,axes=plt.subplots(1,3,figsize=(14.5,4.7),constrained_layout=True)
    ax=axes[0];x=np.arange(len(PAIRS));width=.11;offset=0
    schemes=['State balanced','Uniform spot']
    for mi,m in enumerate(MAPPINGS):
        for si,scheme in enumerate(schemes):
            s=weighting[(weighting.mapping==m)&(weighting.weighting==scheme)].set_index('time_pair').reindex(PAIRS)
            pos=x+(mi*2+si-2.5)*width;ax.bar(pos,s.residual_information,width*.9,color=COLORS[m],alpha=1 if si==0 else .35,hatch='' if si==0 else '//')
    ax.set_xticks(x,_pair_labels(PAIRS),rotation=20,ha='right');ax.set_ylabel('Closure leakage (bit)');ax.grid(axis='y');_panel(ax,'A','Intervention-weighting sensitivity')
    ax.legend(handles=[Line2D([0],[0],color=NAVY,lw=6,label='State balanced'),Line2D([0],[0],color=NAVY,lw=6,alpha=.35,label='Uniform spot')],fontsize=7)
    ax=axes[1];sub=bootstrap[bootstrap.metric=='closure_leakage_bits'];x=np.arange(len(PAIRS));width=.24
    for mi,m in enumerate(MAPPINGS):
        s=sub[sub.mapping==m].set_index('time_pair').reindex(PAIRS);pos=x+(mi-1)*width
        ax.errorbar(pos,s['mean'],yerr=[s['mean']-s['lower_95'],s['upper_95']-s['mean']],fmt=MARKERS[m],color=COLORS[m],capsize=3,label=m)
    ax.set_xticks(x,_pair_labels(PAIRS),rotation=20,ha='right');ax.set_ylabel('Closure leakage (bit)');ax.grid(axis='y');ax.legend(fontsize=6.8);_panel(ax,'B','Stratified bootstrap uncertainty')
    ax=axes[2];x=np.arange(len(soft_hard));w=.34
    ax.bar(x-w/2,soft_hard.hard_primary_induced_js,w,color=BLUE,label='Hard macro primary')
    ax.bar(x+w/2,soft_hard.soft_coordinate_induced_js,w,color=RED,alpha=.72,label='Soft-coordinate diagnostic')
    ax.set_xticks(x,_pair_labels(soft_hard.time_pair),rotation=20,ha='right');ax.set_ylabel('Induced/oracle mean JS');ax.grid(axis='y');ax.legend(fontsize=7);_panel(ax,'C','Optimized soft-vs-hard fairness audit')
    return _save(fig,out,'closure_robustness_and_soft_hard_sensitivity')


def render_paper_closure_figures(*,full,source,lump_summary,lump_states,weighting,horizon,bootstrap,pareto,soft_hard,output_dir):
    return [plot_core_audit(full,lump_summary,output_dir),plot_pareto(pareto,output_dir),plot_horizon(horizon,output_dir),plot_state_failure(source,lump_states,output_dir),plot_robustness(weighting,bootstrap,soft_hard,output_dir)]

