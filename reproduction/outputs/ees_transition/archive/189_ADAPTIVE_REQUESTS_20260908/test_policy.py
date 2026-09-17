"""Independent scalar Exp3 calculation, local-information and timing tests."""
from pathlib import Path
import json, math, numpy as np
from participant_policy import LocalResponse,random_draws,ACTIONS,EXPLORATION
A=Path(__file__).resolve().parent

def main():
    cap=np.array([.005,.010,.015]);q=LocalResponse(cap);weights=np.ones((3,5))
    rng=np.random.default_rng(189);n=0
    for t in range(600):
        base=rng.random(3)*cap;u=rng.random(3);raw,actions,p=q.choose(base,u)
        for i in range(3):
            expected=np.array([(1-EXPLORATION)*w/sum(weights[i])+EXPLORATION/5 for w in weights[i]])
            assert np.max(np.abs(p[i]-expected))<2e-13
            a=next(j for j in range(5) if u[i]<sum(expected[:j+1]))
            assert actions[i]==a and abs(raw[i]-min(base[i]*ACTIONS[a],cap[i]))<1e-14
        y=raw*rng.random(3);q.observe(y)
        for i in range(3):
            weights[i,actions[i]]*=math.exp(EXPLORATION*(y[i]/cap[i])/(5*p[i,actions[i]]))
            weights[i]/=max(weights[i])
        n+=3
    # Credit only the chosen participant/action; no cross-participant update.
    one=LocalResponse(cap);two=LocalResponse(cap);raw,_,_=one.choose(cap,np.array([.1,.3,.8]));two.choose(cap,np.array([.1,.3,.8]))
    one.observe(raw*.5);changed=raw*.5;changed[0]*=.2;two.observe(changed)
    assert np.array_equal(one.probabilities()[1:],two.probabilities()[1:])
    assert not np.array_equal(one.probabilities()[0],two.probabilities()[0])
    # Selected actions genuinely can change admitted input, unlike mere ceiling clipping.
    q=LocalResponse(cap);raw,_,_=q.choose(cap*.8,np.zeros(3));assert np.any(raw<cap*.8)
    assert np.any(np.minimum(raw,cap*.7)!=np.minimum(cap*.8,cap*.7));q.observe(raw*.4)
    changed_probability=np.max(np.abs(q.probabilities()-.2));assert changed_probability>0
    s=LocalResponse(cap,'STATIC_RANDOM')
    for _ in range(100):
        r,_,_=s.choose(cap,rng.random(3));s.observe(r)
    assert np.array_equal(s.state(),np.zeros((3,5)))
    restored=LocalResponse(cap);restored.restore(q.state(),q.completed)
    assert np.array_equal(restored.probabilities(),q.probabilities())
    assert np.array_equal(random_draws(2001,'2013-01-02',0,3),random_draws(2001,'2013-01-02',0,3))
    assert not np.array_equal(random_draws(2001,'2013-01-02',0,3),random_draws(2002,'2013-01-02',0,3))
    # Explicitly reject temporal leakage channels and invalid state transitions.
    rejected=[]
    def rejects(name,fn):
        try:fn()
        except (ValueError,RuntimeError,TypeError):rejected.append(name)
        else:raise AssertionError(name)
    rejects('unobserved_future_availability_argument',lambda:LocalResponse(cap).choose(cap,np.zeros(3),future_available=cap))
    rejects('neighbor_requests_argument',lambda:LocalResponse(cap).choose(cap,np.zeros(3),neighbor_requests=cap))
    rejects('reward_before_choice',lambda:LocalResponse(cap).observe(cap))
    b=LocalResponse(cap);b.choose(cap,np.zeros(3))
    rejects('second_choice_before_delivery',lambda:b.choose(cap,np.zeros(3)))
    rejects('delivery_above_chosen_request',lambda:b.observe(cap))
    rejects('checkpoint_before_delivery',b.state)
    rejects('negative_capacity',lambda:LocalResponse([-1]))
    rejects('nan_capacity',lambda:LocalResponse([float('nan')]))
    rejects('invalid_probability_draw',lambda:LocalResponse(cap).choose(cap,np.ones(3)))
    rejects('request_above_capacity',lambda:LocalResponse(cap).choose(cap*2,np.zeros(3)))
    rejects('static_learned_checkpoint',lambda:LocalResponse(cap,'STATIC_RANDOM').restore(np.ones((3,5)),1))
    q=LocalResponse(cap);q.choose(np.zeros(3),np.zeros(3));q.observe(np.zeros(3));assert np.array_equal(q.state(),np.zeros((3,5)))
    report=dict(passed=True,scalar_participant_updates=n,negative_tests=rejected,
      actual_admitted_request_can_change=True,selected_probability_changes=changed_probability,
      own_delivery_only=True,static_random_unchanged=True,checkpoint_exact=True,
      source='Auer et al., SIAM J. Comput. 32(1), 48–77, Fig. 1, DOI 10.1137/S0097539701398375',
      scope='Implementation tests only; no network performance conclusion or behavioral theorem transfer.')
    (A/'POLICY_TESTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
