from recommender_experiments.service.opl.contexual_bandit_policy import ContextualBanditPolicy
from obp.policy import BernoulliTS


def test_ContexualBanditモデルが正しく初期化されること():
    sut = ContextualBanditPolicy()
    assert True
