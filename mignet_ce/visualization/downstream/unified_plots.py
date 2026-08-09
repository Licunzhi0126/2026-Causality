from __future__ import annotations

"""Publication-style plots for the unified downstream suite.

Every core downstream figure compares the same three representations:
Seurat K150, Seurat K40, and Optimized coarse-graining.
All labels inside figures are English and no figure numbering is embedded.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import (
    NAVY, BLUE, RED, CYAN, GOLD, GREEN, PURPLE, LIGHT, DARK, MUTED, GRID,
    DIVERGING, SEQUENTIAL, set_publication_style, savefig, add_panel_label,
)
from .unified_suite import MAPPINGS, COLORS, MARKERS

PAIR_ORDER = ["11.5->12.5", "12.5->13.5", "13.5->14.5"]
PAIR_LABEL = {p:p.replace("->", "→") for p in PAIR_ORDER}
MAP_SHORT = {"Seurat K150":"K150", "Seurat K40":"K40", "Optimized coarse-graining":"Optimized"}


def _figsize(cols=3, rows=2): return (4.4*cols, 3.45*rows)
def _finish(fig, title=None):
    if title:
        fig.suptitle(title, x=.02, ha='left', fontsize=15.5, fontweight='bold', color=NAVY, y=.995)
    fig.tight_layout(rect=[0,0,1,.965] if title else None)

def _legend_mappings(ax, loc='best'):
    handles=[Line2D([0],[0], marker=MARKERS[m], color='none', markerfacecolor=COLORS[m], markeredgecolor='white', markersize=7, label=MAP_SHORT[m]) for m in MAPPINGS]
    ax.legend(handles=handles, loc=loc)

def _mapping_x(ax):
    ax.set_xticks(range(3), [MAP_SHORT[m] for m in MAPPINGS], rotation=0)

def _ordered(df, pair=None):
    x=df.copy()
    x['mapping']=pd.Categorical(x.mapping, MAPPINGS, ordered=True)
    if pair is not None: x=x[x.time_pair==pair]
    return x.sort_values('mapping')


def plot_causal_emergence_decomposition(t, out):
    set_publication_style(); df=t['metrics']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    # A: delta EI grouped by time pair
    ax=axs[0,0]; x=np.arange(3); width=.22
    for j,m in enumerate(MAPPINGS):
        g=df[df.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); ax.bar(x+(j-1)*width,g.delta_EI_vs_spot,width*.92,color=COLORS[m],label=MAP_SHORT[m])
    ax.axhline(0,color=MUTED,lw=.8); ax.set_xticks(x,[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('ΔEI vs spot (bits)'); ax.set_title('Causal emergence relative to spot'); ax.grid(axis='y'); ax.legend(ncol=3,loc='best'); add_panel_label(ax,'A')
    # B/C retained mechanism decomposition by mapping/pair
    for col,(metric,ylabel,title) in enumerate([('H_effect','Effect diversity H(Y) (bits)','Effect diversity'),('H_noise','Conditional noise H(Y|X) (bits)','Transition noise')],start=1):
        ax=axs[0,col]
        for m in MAPPINGS:
            g=df[df.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); ax.plot(range(3),g[metric],marker=MARKERS[m],lw=1.8,color=COLORS[m],label=MAP_SHORT[m])
        ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis='y'); add_panel_label(ax,chr(ord('A')+col));
        if col==2: _legend_mappings(ax)
    # D determinism-degeneracy
    ax=axs[1,0]
    for m in MAPPINGS:
        g=df[df.mapping==m]
        ax.scatter(g.degeneracy,g.determinism,s=55,c=COLORS[m],marker=MARKERS[m],edgecolors='white',linewidths=.6,label=MAP_SHORT[m])
        for _,r in g.iterrows(): ax.annotate(r.time_pair.split('->')[0],(r.degeneracy,r.determinism),xytext=(4,3),textcoords='offset points',fontsize=6.5,color=MUTED)
    lo=min(df.degeneracy.min(),df.determinism.min()); hi=max(df.degeneracy.max(),df.determinism.max()); ax.plot([lo,hi],[lo,hi],'--',lw=.8,color=GRID); ax.set_xlabel('Degeneracy'); ax.set_ylabel('Determinism'); ax.set_title('Determinism–degeneracy plane'); ax.grid(); add_panel_label(ax,'D')
    # E EI absolute
    ax=axs[1,1]
    for m in MAPPINGS:
        g=df[df.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); ax.plot(range(3),g.EI,marker=MARKERS[m],lw=1.8,color=COLORS[m])
    spot=df.groupby('time_pair').spot_EI.first().reindex(PAIR_ORDER); ax.plot(range(3),spot,'--o',color=MUTED,label='Spot'); ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('Effective information (bits)'); ax.set_title('Absolute EI across scales'); ax.grid(axis='y'); ax.legend(); add_panel_label(ax,'E')
    # F summary retained/noise tradeoff
    ax=axs[1,2]
    for m in MAPPINGS:
        g=df[df.mapping==m]; ax.scatter(g.H_noise,g.H_effect,s=40+90*np.clip(g.EI/g.EI.max(),0,1),c=COLORS[m],marker=MARKERS[m],alpha=.9,edgecolors='white',linewidths=.5)
    ax.set_xlabel('Conditional noise H(Y|X) (bits)'); ax.set_ylabel('Effect diversity H(Y) (bits)'); ax.set_title('Mechanistic trade-off'); ax.grid(); add_panel_label(ax,'F'); _legend_mappings(ax)
    _finish(fig,'Causal emergence and information-theoretic decomposition'); savefig(fig,Path(out)/'causal_emergence_decomposition.png')


def plot_state_level_ei_spatial(t,out):
    set_publication_style(); df=t['spatial_spots']; maps=['Spot',*MAPPINGS]; fig,axs=plt.subplots(4,3,figsize=(13.4,13.7),squeeze=False)
    vals=df.state_ei.to_numpy(); vmax=max(float(np.nanquantile(vals,.98)),1e-6); im=None
    for r,m in enumerate(maps):
        for c,pair in enumerate(PAIR_ORDER):
            ax=axs[r,c]; g=df[(df.mapping==m)&(df.time_pair==pair)]; im=ax.scatter(g.x,g.y,c=g.state_ei,s=2.6 if m=='Spot' else 3.0,cmap=SEQUENTIAL,vmin=0,vmax=vmax,rasterized=True); ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); label='Spot' if m=='Spot' else MAP_SHORT[m]; ax.set_title(f'{label} · {PAIR_LABEL[pair]}',fontsize=9.5); add_panel_label(ax,chr(ord('A')+r*3+c))
    fig.subplots_adjust(left=.035,right=.915,bottom=.03,top=.94,wspace=.055,hspace=.15)
    cax=fig.add_axes([.935,.15,.012,.69]); cbar=fig.colorbar(im,cax=cax); cbar.set_label('State-level EI contribution (bits)')
    fig.suptitle('Spatial localization of EI contribution from spot to macro representations',x=.02,ha='left',fontsize=15.5,fontweight='bold',color=NAVY,y=.988)
    savefig(fig,Path(out)/'state_level_ei_spatial.png')


def plot_single_step_dynamics(t,out):
    set_publication_style(); c=t['closure']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    # info budget 3 panels
    for j,p in enumerate(PAIR_ORDER):
        ax=axs[0,j]; g=_ordered(c,p); x=np.arange(3); retained=g.I_retained_bits.to_numpy(); leak=g.closure_leakage_bits.to_numpy(); ax.bar(x,retained,color=[COLORS[m] for m in MAPPINGS],alpha=.95,label='Retained macro information'); ax.bar(x,leak,bottom=retained,color='#DDE4E9',edgecolor=[COLORS[m] for m in MAPPINGS],linewidth=.9,hatch='//',label='Micro leakage'); _mapping_x(ax); ax.set_ylabel('Available future-macro information (bits)'); ax.set_title(PAIR_LABEL[p]); ax.grid(axis='y'); add_panel_label(ax,chr(65+j));
        for xi,row in enumerate(g.itertuples()):
            if row.signal_status!='informative': ax.text(xi,retained[xi]+leak[xi]+.01,'low signal',rotation=90,ha='center',va='bottom',fontsize=6.2,color=MUTED)
        if j==2: ax.legend(loc='upper right')
    # p95, crossfit, operational gap
    specs=[('induced_p95_js','p95 JS','Approximate lumpability tail'),('crossfit_kl_bits','Cross-fit KL (bits)','Held-out source-spot closure'),('q_operational_gap_js','Excess JS','Separately estimated Q gap')]
    for j,(metric,y,title) in enumerate(specs):
        ax=axs[1,j]; x=np.arange(3); width=.22
        for k,m in enumerate(MAPPINGS):
            g=c[c.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); ax.bar(x+(k-1)*width,g[metric],width*.92,color=COLORS[m],label=MAP_SHORT[m])
        ax.set_xticks(x,[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel(y); ax.set_title(title); ax.grid(axis='y'); add_panel_label(ax,chr(68+j));
        if j==2: ax.legend(ncol=1)
    _finish(fig,'Single-step dynamical closure: information, lumpability, and operational realization'); savefig(fig,Path(out)/'single_step_dynamics_overview.png')


def plot_multistep_dynamics(t,out):
    set_publication_style(); h=t['horizon']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    # per mapping info budget across horizon
    for j,m in enumerate(MAPPINGS):
        ax=axs[0,j]; g=h[(h.mapping==m)&(h.start_time.astype(str)=='11.5')].sort_values('horizon'); x=g.horizon.to_numpy(); ret=g.I_retained_bits.to_numpy(); leak=g.closure_leakage_bits.to_numpy(); ax.bar(x,ret,color=COLORS[m],label='Retained'); ax.bar(x,leak,bottom=ret,color='#E1E7EB',edgecolor=COLORS[m],hatch='//',linewidth=.9,label='Leakage'); ax.set_xticks(x,[f'{int(v)} step' if int(v)==1 else f'{int(v)} steps' for v in x]); ax.set_ylabel('Information (bits)'); ax.set_title(MAP_SHORT[m]); ax.grid(axis='y'); add_panel_label(ax,chr(65+j));
        if j==2: ax.legend()
    # CK excess by windows
    ax=axs[1,0]
    for m in MAPPINGS:
        g=h[(h.mapping==m)&h.ck_composition_excess_js.notna()].copy(); labels=[f"{r.start_time}→{r.end_time}" for r in g.itertuples()]; ax.plot(range(len(g)),g.ck_composition_excess_js,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m]);
    if len(g): ax.set_xticks(range(len(g)),labels,rotation=15)
    ax.set_ylabel('CK composition excess JS'); ax.set_title('Long-horizon propagator consistency'); ax.grid(axis='y'); add_panel_label(ax,'D'); ax.legend()
    # total available decay
    ax=axs[1,1]
    for m in MAPPINGS:
        g=h[(h.mapping==m)&(h.start_time.astype(str)=='11.5')].sort_values('horizon'); ax.plot(g.horizon,g.I_available_bits,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.set_xlabel('Prediction horizon'); ax.set_ylabel('Available information (bits)'); ax.set_title('Predictive signal decay'); ax.grid(); add_panel_label(ax,'E')
    # retained fraction informative only
    ax=axs[1,2]
    for m in MAPPINGS:
        g=h[(h.mapping==m)&(h.start_time.astype(str)=='11.5')].sort_values('horizon'); q=np.divide(g.I_retained_bits,g.I_available_bits,out=np.full(len(g),np.nan),where=g.I_available_bits>.02); ax.plot(g.horizon,q,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.set_ylim(0,1.05); ax.set_xlabel('Prediction horizon'); ax.set_ylabel('Retained / available'); ax.set_title('Non-trivial closure across horizons'); ax.grid(); add_panel_label(ax,'F'); _legend_mappings(ax)
    _finish(fig,'Multi-step predictive closure and Chapman–Kolmogorov consistency'); savefig(fig,Path(out)/'multistep_dynamics_overview.png')


def plot_random_null(t,out):
    set_publication_style(); n=t['null']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    for j,p in enumerate(PAIR_ORDER):
        ax=axs[0,j]
        for m in MAPPINGS:
            rand=n[(n.mapping==m)&(n.time_pair==p)&(n.kind=='matched_random')].EI; obs=n[(n.mapping==m)&(n.time_pair==p)&(n.kind=='observed')].EI.iloc[0];
            if len(rand)>1:
                xs=np.linspace(rand.min(),rand.max(),60); # histogram as step density
                ax.hist(rand,bins=18,density=True,histtype='step',lw=1.25,color=COLORS[m],alpha=.9,label=f'{MAP_SHORT[m]} null')
            ax.axvline(obs,color=COLORS[m],lw=2,alpha=.95)
        ax.set_xlabel('Induced macro EI (bits)'); ax.set_ylabel('Null density'); ax.set_title(PAIR_LABEL[p]); add_panel_label(ax,chr(65+j));
        if j==2: ax.legend(fontsize=6.5)
    # effect size views EI and leakage
    for j,metric in enumerate(['EI','closure_leakage_bits']):
        ax=axs[1,j]
        rows=[]
        for p in PAIR_ORDER:
            for m in MAPPINGS:
                rr=n[(n.mapping==m)&(n.time_pair==p)&(n.kind=='matched_random')][metric]; ob=n[(n.mapping==m)&(n.time_pair==p)&(n.kind=='observed')][metric].iloc[0]; rows.append((m,p,ob,float(rr.mean()),float(rr.std(ddof=1)) if len(rr)>1 else np.nan))
        for m in MAPPINGS:
            gg=[r for r in rows if r[0]==m]; vals=[(r[2]-r[3])/r[4] if np.isfinite(r[4]) and r[4]>0 else np.nan for r in gg]; ax.plot(range(3),vals,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
        ax.axhline(0,color=MUTED,lw=.8); ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('Standardized observed–null effect'); ax.set_title('EI specificity' if metric=='EI' else 'Closure-leakage specificity'); ax.grid(axis='y'); add_panel_label(ax,chr(68+j));
        if j==1: _legend_mappings(ax)
    ax=axs[1,2]; # null rank summary
    for m in MAPPINGS:
        vals=[]
        for p in PAIR_ORDER:
            rr=n[(n.mapping==m)&(n.time_pair==p)&(n.kind=='matched_random')].EI; ob=n[(n.mapping==m)&(n.time_pair==p)&(n.kind=='observed')].EI.iloc[0]; vals.append((1+(rr>=ob).sum())/(1+len(rr)))
        ax.plot(range(3),vals,marker=MARKERS[m],color=COLORS[m],lw=1.8)
    ax.set_yscale('log'); ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('Empirical p-value'); ax.set_title('Matched-partition empirical significance'); ax.grid(axis='y'); add_panel_label(ax,'F'); _legend_mappings(ax)
    _finish(fig,'Matched random coarse-graining null model'); savefig(fig,Path(out)/'matched_random_null.png')


def plot_consistency(t,out):
    set_publication_style(); c=t['consistency']; eff=t['effective']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    pairs=[('Seurat K150','Seurat K40'),('Seurat K150','Optimized coarse-graining'),('Seurat K40','Optimized coarse-graining')]
    for j,(a,b) in enumerate(pairs):
        ax=axs[0,j]; g=c[(c.mapping_a==a)&(c.mapping_b==b)].copy(); g['time']=g['time'].astype(str); g=g.set_index('time').reindex(['11.5','12.5','13.5','14.5']); ax.plot(range(4),g.NMI,'o-',color=BLUE,label='NMI'); ax.plot(range(4),g.ARI,'s--',color=RED,label='ARI'); ax.set_ylim(-.05,1.05); ax.set_xticks(range(4),['11.5','12.5','13.5','14.5']); ax.set_ylabel('Partition agreement'); ax.set_title(f'{MAP_SHORT[a]} vs {MAP_SHORT[b]}'); ax.grid(axis='y'); add_panel_label(ax,chr(65+j));
        if j==2: ax.legend()
    # effective K, Keff, optimized incoming/outgoing
    ax=axs[1,0]
    for m in MAPPINGS:
        g=eff[eff.mapping==m].copy(); g['time']=g['time'].astype(str); g=g.set_index('time').reindex(['11.5','12.5','13.5','14.5']); ax.plot(range(4),g.active_k,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.set_xticks(range(4),['11.5','12.5','13.5','14.5']); ax.set_ylabel('Active states'); ax.set_title('Actual representation size'); ax.grid(axis='y'); add_panel_label(ax,'D'); ax.legend()
    ax=axs[1,1]
    for m in MAPPINGS:
        g=eff[eff.mapping==m].copy(); g['time']=g['time'].astype(str); g=g.set_index('time').reindex(['11.5','12.5','13.5','14.5']); ax.plot(range(4),g.Keff,marker=MARKERS[m],color=COLORS[m],lw=1.8)
    ax.set_xticks(range(4),['11.5','12.5','13.5','14.5']); ax.set_ylabel('Effective state count'); ax.set_title('Usage-weighted effective states'); ax.grid(axis='y'); add_panel_label(ax,'E')
    ax=axs[1,2]; g=c[c.mapping_a=='Optimized incoming'];
    if len(g):
        ax.bar(np.arange(len(g))-.16,g.NMI,.3,color=GREEN,label='NMI'); ax.bar(np.arange(len(g))+.16,g.ARI,.3,color=PURPLE,label='ARI'); ax.set_xticks(range(len(g)),g.time.astype(str))
    ax.set_ylim(-.05,1.05); ax.set_ylabel('Agreement'); ax.set_title('Optimized state stability across adjacent fits'); ax.grid(axis='y'); ax.legend(); add_panel_label(ax,'F')
    _finish(fig,'Cross-representation consistency and effective state usage'); savefig(fig,Path(out)/'cross_representation_consistency.png')


def plot_effective_spatial(t,out):
    set_publication_style(); e=t['effective']; s=t['spatial']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    metrics=[('Keff','Effective state count','Effective state usage'),('max_usage','Largest-state usage','State occupancy concentration'),('assignment_confidence','Mean assignment confidence','Assignment certainty')]
    for j,(metric,y,title) in enumerate(metrics):
        ax=axs[0,j]
        for m in MAPPINGS:
            g=e[e.mapping==m].set_index('time').reindex(['11.5','12.5','13.5','14.5']); ax.plot(range(4),g[metric],marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
        ax.set_xticks(range(4),['11.5','12.5','13.5','14.5']); ax.set_ylabel(y); ax.set_title(title); ax.grid(axis='y'); add_panel_label(ax,chr(65+j));
        if j==2: _legend_mappings(ax)
    specs=[('fragmentation','Mean fragmentation','Spatial fragmentation'),('boundary_ratio','Mean boundary ratio','Boundary exposure'),('moran_i',"Mean Moran's I",'Spatial autocorrelation')]
    for j,(metric,y,title) in enumerate(specs):
        ax=axs[1,j]; agg=s.groupby(['mapping','time_pair'])[metric].mean().reset_index()
        for m in MAPPINGS:
            g=agg[agg.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); ax.plot(range(3),g[metric],marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
        ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel(y); ax.set_title(title); ax.grid(axis='y'); add_panel_label(ax,chr(68+j));
        if j==2: _legend_mappings(ax)
    _finish(fig,'Effective states and spatial morphology'); savefig(fig,Path(out)/'effective_states_spatial.png')


def plot_mechanism(t,out):
    set_publication_style(); d=t['mechanism']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    # correlations by pair / mapping
    specs=[('grn_concentration','GRN concentration'),('cci_out_log','CCI out-strength'),('cci_in_log','CCI in-strength')]
    for j,(xcol,xlab) in enumerate(specs):
        ax=axs[0,j]
        for m in MAPPINGS:
            vals=[]
            for p in PAIR_ORDER:
                g=d[(d.mapping==m)&(d.time_pair==p)]; r=spearmanr(g[xcol],g.state_ei,nan_policy='omit').statistic if len(g)>2 else np.nan; vals.append(r)
            ax.plot(range(3),vals,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
        ax.axhline(0,color=MUTED,lw=.8); ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylim(-1,1); ax.set_ylabel('Spearman ρ with state EI'); ax.set_title(xlab); ax.grid(axis='y'); add_panel_label(ax,chr(65+j));
        if j==2: _legend_mappings(ax)
    # pooled scatter panels
    for j,(xcol,xlab) in enumerate(specs):
        ax=axs[1,j]
        for m in MAPPINGS:
            g=d[d.mapping==m]; ax.scatter(g[xcol],g.state_ei,s=10,alpha=.5,c=COLORS[m],marker=MARKERS[m],label=MAP_SHORT[m])
        ax.set_xlabel(xlab); ax.set_ylabel('State EI (bits)'); ax.set_title(f'{xlab} vs state EI'); ax.grid(); add_panel_label(ax,chr(68+j));
        if j==2: _legend_mappings(ax)
    _finish(fig,'GRN–CCI mechanism audit on a common spot-level substrate'); savefig(fig,Path(out)/'grn_cci_mechanism.png')


def plot_fate(t,out):
    set_publication_style(); d=t['fate']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    # distributions per mapping
    for j,m in enumerate(MAPPINGS):
        ax=axs[0,j]; g=d[d.mapping==m]; sc=ax.scatter(g.first_branch_entropy,g.path_probability,c=g.source_ei,cmap=SEQUENTIAL,s=18,alpha=.75,edgecolors='none'); ax.set_xlabel('First-step branch entropy (bits)'); ax.set_ylabel('Main-path probability'); ax.set_title(MAP_SHORT[m]); ax.grid(); add_panel_label(ax,chr(65+j));
        if j==2: fig.colorbar(sc,ax=ax,fraction=.046,pad=.03,label='Source state EI')
    # endpoint entropy / relation
    ax=axs[1,0]
    for m in MAPPINGS:
        g=d[d.mapping==m]; ax.scatter(g.source_ei,g.endpoint_entropy,s=15,alpha=.55,c=COLORS[m],marker=MARKERS[m],label=MAP_SHORT[m])
    ax.set_xlabel('Source state EI (bits)'); ax.set_ylabel('Endpoint entropy (bits)'); ax.set_title('EI and terminal fate concentration'); ax.grid(); add_panel_label(ax,'D'); ax.legend()
    ax=axs[1,1]
    vals=[]
    for m in MAPPINGS:
        g=d[d.mapping==m]; vals.append([g.path_probability.mean(),g.first_branch_entropy.mean(),g.endpoint_entropy.mean()])
    x=np.arange(3); w=.22
    for k,m in enumerate(MAPPINGS): ax.bar(x+(k-1)*w,vals[k],w*.92,color=COLORS[m],label=MAP_SHORT[m])
    ax.set_xticks(x,['Path prob.','Branch entropy','Endpoint entropy']); ax.set_title('Fate-graph summary'); ax.grid(axis='y'); add_panel_label(ax,'E')
    ax=axs[1,2]
    for m in MAPPINGS:
        g=d[d.mapping==m]; q=np.quantile(g.path_probability,[.1,.25,.5,.75,.9]); ax.plot([.1,.25,.5,.75,.9],q,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.set_xlabel('Quantile'); ax.set_ylabel('Main-path probability'); ax.set_title('Path-stability distribution'); ax.grid(); add_panel_label(ax,'F'); _legend_mappings(ax)
    _finish(fig,'Long-range macro fate paths across representations'); savefig(fig,Path(out)/'macro_fate_paths.png')


def plot_perturbation(t,out):
    set_publication_style(); d=t['perturbation']; fig,axs=plt.subplots(2,3,figsize=_figsize())
    # rows mapping, columns? use pair avg per mapping targeted lines
    targets=['high_state_ei','high_cci_out','high_grn_concentration','matched_random']; tc={'high_state_ei':RED,'high_cci_out':BLUE,'high_grn_concentration':GREEN,'matched_random':MUTED}; tl={'high_state_ei':'High state EI','high_cci_out':'High CCI out','high_grn_concentration':'High GRN concentration','matched_random':'Matched random'}
    for j,m in enumerate(MAPPINGS):
        ax=axs[0,j]; g=d[d.mapping==m].groupby(['target','dose']).ei_drop_mean.mean().reset_index()
        for target in targets:
            z=g[g.target==target]; ax.plot(z.dose,z.ei_drop_mean,marker='o',ms=3.5,lw=1.6,color=tc[target],label=tl[target])
        ax.axhline(0,color=GRID,lw=.8); ax.set_xlabel('Perturbation dose'); ax.set_ylabel('Mean EI decrease (bits)'); ax.set_title(MAP_SHORT[m]); ax.grid(); add_panel_label(ax,chr(65+j));
        if j==2: ax.legend(fontsize=6.5)
    # targeted over random at full dose per pair
    for j,target in enumerate(targets[:3]):
        ax=axs[1,j]; x=np.arange(3); w=.22
        for k,m in enumerate(MAPPINGS):
            vals=[]
            for p in PAIR_ORDER:
                tar=d[(d.mapping==m)&(d.time_pair==p)&(d.target==target)&np.isclose(d.dose,1)].ei_drop_mean.iloc[0]; rnd=d[(d.mapping==m)&(d.time_pair==p)&(d.target=='matched_random')&np.isclose(d.dose,1)].ei_drop_mean.iloc[0]; vals.append(tar-rnd)
            ax.bar(x+(k-1)*w,vals,w*.92,color=COLORS[m],label=MAP_SHORT[m])
        ax.axhline(0,color=MUTED,lw=.8); ax.set_xticks(x,[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('Targeted − random EI decrease'); ax.set_title(tl[target]); ax.grid(axis='y'); add_panel_label(ax,chr(68+j));
        if j==2: _legend_mappings(ax)
    _finish(fig,'Virtual perturbation response across all representations'); savefig(fig,Path(out)/'virtual_perturbation.png')

# ---------------- closure unified five ----------------

def plot_closure_information_budget_operational(t,out):
    set_publication_style(); c=t['closure']; fig,axs=plt.subplots(1,3,figsize=(13.4,3.9))
    ax=axs[0]; x=np.arange(3); w=.22
    # group bars each mapping with retained/leak stacked
    for k,m in enumerate(MAPPINGS):
        g=c[c.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); xpos=x+(k-1)*w; ax.bar(xpos,g.I_retained_bits,w*.9,color=COLORS[m],label=MAP_SHORT[m]); ax.bar(xpos,g.closure_leakage_bits,w*.9,bottom=g.I_retained_bits,color='#E5EAEF',edgecolor=COLORS[m],hatch='//',linewidth=.7)
    ax.set_xticks(x,[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('Future-macro information (bits)'); ax.set_title('Information budget: retained + micro leakage'); ax.grid(axis='y'); add_panel_label(ax,'A'); ax.legend(ncol=3)
    ax=axs[1]
    for m in MAPPINGS:
        g=c[c.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); ax.plot(range(3),g.induced_p95_js,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('p95 intrinsic JS'); ax.set_title('Approximate-lumpability tail'); ax.grid(axis='y'); add_panel_label(ax,'B'); _legend_mappings(ax)
    ax=axs[2]
    for m in MAPPINGS:
        g=c[c.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); ax.scatter(g.crossfit_kl_bits,g.q_operational_gap_js,c=COLORS[m],marker=MARKERS[m],s=60,label=MAP_SHORT[m],edgecolors='white',linewidths=.6)
        for idx,r in g.iterrows(): ax.annotate(str(idx).split('->')[0],(r.crossfit_kl_bits,r.q_operational_gap_js),xytext=(4,3),textcoords='offset points',fontsize=6.5,color=MUTED)
    ax.set_xlabel('Cross-fit KL (bits; lower better)'); ax.set_ylabel('Separately estimated Q gap (JS)'); ax.set_title('Operational realization'); ax.grid(); add_panel_label(ax,'C'); _legend_mappings(ax)
    _finish(fig,'Core dynamical-closure audit'); savefig(fig,Path(out)/'closure_information_budget_and_operational_q.png')


def plot_closure_pareto(t,out):
    set_publication_style(); p=t['pareto']; fig,axs=plt.subplots(1,3,figsize=(13.6,4.05))
    # color candidates by retained info, anchors recognizable
    for pair,alpha in zip(PAIR_ORDER,[1,.75,.52]):
        g=p[p.time_pair==pair]
        cand=g[g.mapping=='Optimized candidate'];
        if len(cand):
            sc=axs[0].scatter(cand.canonical_delta_ei,cand.closure_quality,c=cand.I_retained_bits,cmap=SEQUENTIAL,s=24+2*cand.active_k,alpha=alpha,edgecolors='none')
        for m in MAPPINGS:
            a=g[g.mapping==m]
            if len(a): axs[0].scatter(a.canonical_delta_ei,a.closure_quality,c=COLORS[m],marker=MARKERS[m],s=82,edgecolors='white',linewidths=.8,zorder=5)
    axs[0].set_xlabel('Canonical ΔEI vs spot (bits)'); axs[0].set_ylabel('Closure quality: retained / available'); axs[0].set_ylim(-.02,1.02); axs[0].set_title('Causal emergence vs closure'); axs[0].grid(); add_panel_label(axs[0],'A')
    # panel B predictive compression
    for pair,alpha in zip(PAIR_ORDER,[1,.75,.52]):
        g=p[p.time_pair==pair]; cand=g[g.mapping=='Optimized candidate'];
        if len(cand): axs[1].scatter(cand.compression_gain,cand.I_retained_bits,c=cand.closure_quality,cmap=SEQUENTIAL,s=24+2*cand.active_k,alpha=alpha,edgecolors='none')
        for m in MAPPINGS:
            a=g[g.mapping==m];
            if len(a): axs[1].scatter(a.compression_gain,a.I_retained_bits,c=COLORS[m],marker=MARKERS[m],s=82,edgecolors='white',linewidths=.8,zorder=5)
    axs[1].set_xlabel('Compression gain'); axs[1].set_ylabel('Retained predictive information (bits)'); axs[1].set_title('Predictive compression'); axs[1].grid(); add_panel_label(axs[1],'B')
    # operational only Pareto-front candidates and anchors
    ax=axs[2]
    g=p[p.front_emergence_closure|p.front_compression_prediction]
    for m in ['Optimized candidate']+list(MAPPINGS):
        z=g[g.mapping==m]
        if not len(z): continue
        color=GREEN if m=='Optimized candidate' else COLORS[m]; marker='o' if m=='Optimized candidate' else MARKERS[m]; label='Optimized candidates' if m=='Optimized candidate' else MAP_SHORT[m]
        ax.scatter(z.crossfit_kl_bits,z.q_operational_gap_js,c=color,marker=marker,s=35 if m=='Optimized candidate' else 82,alpha=.7 if m=='Optimized candidate' else 1,edgecolors='white',linewidths=.5,label=label)
    ax.set_xlabel('Cross-fit leakage KL (bits)'); ax.set_ylabel('Operational Q gap (JS)'); ax.set_title('Operational audit of non-dominated candidates'); ax.grid(); add_panel_label(ax,'C'); ax.legend(fontsize=6.5)
    _finish(fig,'Causal-emergence–closure Pareto candidate landscape'); savefig(fig,Path(out)/'closure_causal_emergence_pareto.png')


def plot_closure_horizon(t,out):
    set_publication_style(); h=t['horizon']; fig,axs=plt.subplots(1,2,figsize=(9.2,3.9))
    ax=axs[0]; # three representations each as two stacked bands per horizon via grouped bars
    base=np.arange(3); w=.22
    for k,m in enumerate(MAPPINGS):
        g=h[(h.mapping==m)&(h.start_time.astype(str)=='11.5')].sort_values('horizon'); xpos=base+(k-1)*w; ax.bar(xpos,g.I_retained_bits,w*.9,color=COLORS[m],label=MAP_SHORT[m]); ax.bar(xpos,g.closure_leakage_bits,w*.9,bottom=g.I_retained_bits,color='#E5EAEF',edgecolor=COLORS[m],hatch='//',linewidth=.7)
    ax.set_xticks(base,['1 step','2 steps','3 steps']); ax.set_ylabel('Information (bits)'); ax.set_title('Horizon information budget'); ax.grid(axis='y'); add_panel_label(ax,'A'); ax.legend(ncol=3)
    ax=axs[1]
    for m in MAPPINGS:
        g=h[(h.mapping==m)&h.ck_composition_excess_js.notna()].copy(); labels=[f'{r.start_time}→{r.end_time}' for r in g.itertuples()]; ax.plot(range(len(g)),g.ck_composition_excess_js,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    if len(g): ax.set_xticks(range(len(g)),labels)
    ax.set_ylabel('CK composition excess JS'); ax.set_title('Chapman–Kolmogorov consistency'); ax.grid(axis='y'); add_panel_label(ax,'B'); _legend_mappings(ax)
    _finish(fig,'Long-horizon information closure and propagator composition'); savefig(fig,Path(out)/'closure_horizon_ck_consistency.png')


def plot_closure_failure_repair(t,out):
    set_publication_style(); st=t['leakage_states']; spx=t['leakage_spots']; rep=t['repairability']; fig,axs=plt.subplots(1,3,figsize=(13.6,4.05))
    ax=axs[0]
    for m in MAPPINGS:
        g=st[st.mapping==m].copy(); # average ranked shares across pairs after rank
        allshares=[]
        for p in PAIR_ORDER:
            z=g[g.time_pair==p].sort_values('leakage_bits',ascending=False); cs=z.leakage_bits.cumsum()/max(z.leakage_bits.sum(),1e-12); xx=np.arange(1,len(z)+1)/len(z); allshares.append((xx,cs.to_numpy()))
        # interpolate common grid
        grid=np.linspace(0,1,101); vals=[]
        for xx,yy in allshares: vals.append(np.interp(grid,np.r_[0,xx],np.r_[0,yy]))
        ax.plot(grid,np.mean(vals,axis=0),color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.plot([0,1],[0,1],'--',color=GRID,lw=1); ax.set_xlabel('Fraction of states (ranked by leakage)'); ax.set_ylabel('Cumulative leakage share'); ax.set_title('Closure failure concentration'); ax.grid(); add_panel_label(ax,'A'); _legend_mappings(ax)
    ax=axs[1]; # spatial hotspot overlap: gray tissue background + top leakage spots per representation
    p=PAIR_ORDER[0]; base=spx[(spx.mapping==MAPPINGS[0])&(spx.time_pair==p)].copy(); ax.scatter(base.x,base.y,s=2,c='#DDE3E7',alpha=.65,rasterized=True)
    for m in MAPPINGS:
        g=spx[(spx.mapping==m)&(spx.time_pair==p)].copy(); thr=g.spot_leakage_kl.quantile(.95); z=g[g.spot_leakage_kl>=thr]; ax.scatter(z.x,z.y,s=14,c=COLORS[m],marker=MARKERS[m],alpha=.72,label=f'{MAP_SHORT[m]} top 5%',edgecolors='white',linewidths=.25,rasterized=True)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'Top leakage hotspots · {PAIR_LABEL[p]}'); add_panel_label(ax,'B'); ax.legend(fontsize=6.2,loc='best')
    ax=axs[2]
    for m in MAPPINGS:
        g=rep[rep.mapping==m].groupby('step').agg(leakage=('leakage_bits','mean'),rand=('random_gain_mean','mean')).reset_index(); ax.plot(g.step,g.leakage,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.set_xlabel('Targeted oracle splits'); ax.set_ylabel('Primary closure leakage (bits)'); ax.set_title('Oracle repairability under matched semantics'); ax.grid(); add_panel_label(ax,'C'); _legend_mappings(ax)
    _finish(fig,'Closure-failure localization and oracle repairability'); savefig(fig,Path(out)/'closure_failure_localization_and_repairability.png')


def plot_closure_robustness(t,out):
    set_publication_style(); wdf=t['weighting']; bdf=t['bootstrap']; sh=t['soft_hard']; nul=t['null']; eff=t['effective']; fig,axs=plt.subplots(2,2,figsize=(9.5,7.2))
    ax=axs[0,0]
    for m in MAPPINGS:
        z=wdf[wdf.mapping==m]; sb=z[z.weighting=='state_balanced'].set_index('time_pair').reindex(PAIR_ORDER); us=z[z.weighting=='uniform_spot'].set_index('time_pair').reindex(PAIR_ORDER); ax.plot(range(3),us.closure_leakage_bits-sb.closure_leakage_bits,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.axhline(0,color=MUTED,lw=.8); ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('Uniform − balanced leakage (bits)'); ax.set_title('Intervention-weighting sensitivity'); ax.grid(axis='y'); add_panel_label(ax,'A'); _legend_mappings(ax)
    ax=axs[0,1]; x=np.arange(3); ww=.22
    for k,m in enumerate(MAPPINGS):
        g=bdf[bdf.mapping==m].set_index('time_pair').reindex(PAIR_ORDER); y=g['mean'].to_numpy(); lo=y-g.low.to_numpy(); hi=g.high.to_numpy()-y; ax.errorbar(x+(k-1)*ww,y,yerr=np.vstack([lo,hi]),fmt=MARKERS[m],ms=5,color=COLORS[m],capsize=2,lw=1.2,label=MAP_SHORT[m])
    ax.set_xticks(x,[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('Bootstrap closure leakage (bits)'); ax.set_title('Sampling uncertainty'); ax.grid(axis='y'); add_panel_label(ax,'B'); _legend_mappings(ax)
    ax=axs[1,0]
    if len(sh):
        x=np.arange(len(sh)); ax.bar(x-.17,sh.hard_primary_js,.32,color=GREEN,label='Hard-state primary'); ax.bar(x+.17,sh.soft_coordinate_js,.32,color=PURPLE,label='Soft-coordinate sensitivity'); ax.set_xticks(x,[PAIR_LABEL[p] for p in sh.time_pair])
    ax.set_ylabel('Intrinsic mean JS'); ax.set_title('Soft–hard fairness sensitivity'); ax.grid(axis='y'); add_panel_label(ax,'C'); ax.legend()
    ax=axs[1,1]
    # observed percentile among null plus active K label
    for m in MAPPINGS:
        vals=[]
        for p in PAIR_ORDER:
            rr=nul[(nul.mapping==m)&(nul.time_pair==p)&(nul.kind=='matched_random')].EI; ob=nul[(nul.mapping==m)&(nul.time_pair==p)&(nul.kind=='observed')].EI.iloc[0]; vals.append(100*np.mean(rr<ob))
        ax.plot(range(3),vals,marker=MARKERS[m],color=COLORS[m],lw=1.8,label=MAP_SHORT[m])
    ax.set_ylim(0,102); ax.set_xticks(range(3),[PAIR_LABEL[p] for p in PAIR_ORDER]); ax.set_ylabel('Observed EI percentile in matched null (%)'); ax.set_title('Matched-null specificity and actual compression'); ax.grid(axis='y'); add_panel_label(ax,'D'); _legend_mappings(ax)
    # annotate optimized active K
    opt=eff[eff.mapping=='Optimized coarse-graining'].copy(); opt['time']=opt['time'].astype(str); opt=opt.set_index('time');
    for i,p in enumerate(PAIR_ORDER):
        tt=p.split('->')[0]
        if tt in opt.index: ax.text(i,4,f"active K={int(opt.loc[tt,'active_k'])}\nconf={opt.loc[tt,'assignment_confidence']:.2f}",ha='center',va='bottom',fontsize=6.4,color=GREEN)
    _finish(fig,'Robustness, fairness, and matched-null audit'); savefig(fig,Path(out)/'closure_robustness_fairness_and_null.png')


def render_all_unified_figures(tables, output_root:Path):
    out=Path(output_root)/'figures'; out.mkdir(parents=True,exist_ok=True)
    funcs=[
        plot_causal_emergence_decomposition,plot_state_level_ei_spatial,plot_single_step_dynamics,plot_multistep_dynamics,
        plot_random_null,plot_consistency,plot_effective_spatial,plot_mechanism,plot_fate,plot_perturbation,
        plot_closure_information_budget_operational,plot_closure_pareto,plot_closure_horizon,plot_closure_failure_repair,plot_closure_robustness,
    ]
    for f in funcs: f(tables,out)
    return sorted(out.glob('*.png'))

