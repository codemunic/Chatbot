Simple Chatbot for Restaurant enquiry.

Sample inputs and corressponding responses are given in json file.

Model classify the input into one of the category from json file and then choose a random response corresponding to that category.

Code written in Pytorch.

Run:

Python train.py
python chat.py

Working sample:

Let's chat! (type 'quit' to exit)
You: Hello
Bot: Hi there, how can I help?
You: What can you do?
Bot: We sell coffee and tea
You: What is the cost of that?
Bot: We provide coffee and tea at reasonable price.
You: How can I pay for this?
Bot: We accept VISA, Mastercard and Paypal
You: How much time do you take for delivery?
Bot: Delivery takes 2-4 days
You: Okay. Thanks for help
Bot: My pleasure
You: Bye
Bot: Bye! Come back again soon.
You: quit 

References:

https://github.com/python-engineer/pytorch-chatbot