import logging
from Model import Model

# Init for logging
logging.basicConfig(level = logging.INFO)
#logging.basicConfig(level = logging.DEBUG)

if __name__ == '__main__':
    model = Model(force_cpu = True)
    prompt = "Name the provinces of Canada"
    logging.info("Prompt: " + prompt)
    logging.info("The model is computing a reply...")
    reply = model.getPromptReply(prompt)
    logging.info("Reply: " + reply)