
import pytest
import sys, os
sys.path.append(os.path.join(sys.path[0],'src'))
from Model import Model
 
def test_llmReplyFound():
    model = Model(force_cpu = True)
    prompt = "Name the provinces of Canada"
    reply = model.getPromptReply(prompt)
    assert "Ontario" in reply
