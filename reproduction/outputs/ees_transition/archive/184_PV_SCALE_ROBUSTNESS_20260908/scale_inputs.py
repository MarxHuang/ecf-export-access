"""Reconstruct exogenous scale scenarios directly from fixed profile data."""
from pathlib import Path
import csv,json
import numpy as np

def near(a,b,tol=1e-10):
    delta=np.asarray(a)-np.asarray(b)
    assert np.isfinite(delta).all()
    err=float(np.max(np.abs(delta)))
    assert err<tol,(err,tol)

class InputReference:
    def __init__(self,p):
        from run_scale import load_runner
        m=load_runner();self.p=p;self.base_scale=m.PV_SCALE
        cfg=p['config'];root=Path(cfg['package'])
        self.current=np.load(root/'AUSGRID_EVALUATION_2012_2013.npz',allow_pickle=False)
        self.previous=np.load(root/'AUSGRID_DEVELOPMENT_2011_2012.npz',allow_pickle=False)
        self.training=np.load(root/'AUSGRID_TRAIN_2010_2011.npz',allow_pickle=False)
        with Path(cfg['mapping_file']).open(encoding='utf-8',newline='') as f:rows={r['load_name']:r for r in csv.DictReader(f)}
        conn=m.parse_dss_load_connections(Path(cfg['dss_root']))
        self.indices=np.array([int(rows[c.load_name]['ausgrid_customer_id'])-1 for c in conn])
        self.part=np.array([rows[c.load_name]['participant_33pct']=='1' for c in conn])
        assert len(self.part)==1255 and self.part.sum()==418
        self.load_scale=float(cfg['load_scale']);self.quantile=m.FORECAST_QUANTILE
        self.cache={};self.dates=list(map(str,self.current['dates']))

    def coefficients(self,factor):
        if factor not in self.cache:
            scale=self.base_scale*factor
            a=np.maximum(scale*self.training['gross_generation_kw']-self.load_scale*self.training['household_load_kw'],0)
            good=self.training['complete_customer_day'] & self.training['chronology_valid_day'][:,None]
            diff=np.diff(a.astype(float),axis=0)
            diff=np.where((good[:-1]&good[1:])[:,None,:],diff,np.nan)
            increment=np.maximum(np.nanquantile(diff,self.quantile,axis=0),0)
            cap=scale*self.current['generator_capacity_kwp'].astype(float)
            assert np.isfinite(increment).all() and np.all(cap>0)
            self.cache[factor]=(cap,increment)
        return self.cache[factor]

    def expected(self,factor,date):
        cap,increment=self.coefficients(factor);scale=self.base_scale*factor;i=self.dates.index(date)
        prev=self.current if i else self.previous;j=i-1 if i else -1
        prior=np.maximum(scale*prev['gross_generation_kw'][j]-self.load_scale*prev['household_load_kw'][j],0)
        raw=np.minimum(prior+increment,cap[None,:])[:,self.indices]/1000
        persistence=np.minimum(prior,cap[None,:])[:,self.indices]/1000
        raw[:,~self.part]=0;persistence[:,~self.part]=0
        load=self.load_scale*self.current['household_load_kw'][i].astype(float)[:,self.indices]
        pv=scale*self.current['gross_generation_kw'][i].astype(float)[:,self.indices]
        # Nonparticipants have household demand but no scenario PV connection.
        available=np.maximum(np.where(self.part,pv,0)-load,0)/1000
        capacity=np.where(self.part,cap[self.indices]/1000,0)
        return dict(capacity=capacity,increment=increment,raw=raw,persistence=persistence,load=load,pv=pv,available=available)

    def verify(self,out,factor,date):
        expected=self.expected(factor,date)
        with np.load(out/'VECTORS.npz') as z:
            assert np.array_equal(z['participant'],self.part)
            near(z['capacity_mw'],expected['capacity'])
            for key,ref in [('load_kw','load'),('pv_kw','pv'),('available_mw','available')]:near(z[key].reshape(48,5,-1)[:,0],expected[ref],1e-9)
            request=z['request_mw'].reshape(48,5,-1)
            near(request[:,0],expected['raw']);near(request[:,1],expected['persistence'])
        return True
