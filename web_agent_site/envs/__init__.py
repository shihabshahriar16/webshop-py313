from web_agent_site.envs.gym_compat import register

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
try:
    # The Selenium-driven env is optional: the text env does not need a browser.
    from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
except ImportError:
    pass

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)
