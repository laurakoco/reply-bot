
import re

greeting_response = ["hi", "hello", "hey"]
goodbye_response = ["bye", "talk to you later"]
thank_response = ['happy to help','don\'t mention it','my pleasure']
inquiry_response = ['i\'m doing ok','ok','i\'ve been better']
future_response = ['yes','no','maybe']
what_you_response = ['nothing', 'not much']
are_you_response = ['yes','no', 'maybe']
when_response = ['soon','not now']
no_response = ['[No Suggestion]']

def bot_response(user_input):

    # pattern search

    # greeting
    if re.match('hi|hi\s+there|hello|hey|good\s+[morning|afternoon|evening|day]', user_input): # greeting
        return greeting_response

    # goodbye
    elif re.match('goodbye|bye|see\s+ya|gotta\s+go', user_input):
        return goodbye_response

    # how are you
    elif re.match('how\s+are\s+you.*|how.*going', user_input):
        return inquiry_response

    # thanks
    elif re.match('thank', user_input):
        return thank_response

    # future (will you, can you, would you, do you)
    elif re.search('[will|can|would|do]\s+you', user_input):
        return future_response

    # are you
    elif re.match('are.*you', user_input):
        return are_you_response

    # when
    elif re.match('when.*you', user_input):
        return when_response

    # what's up
    elif re.match('sup|what.*[happening|up|you]', user_input):
        return what_you_response

    else: # else
        return no_response

print("* Hello! Type in a message and I will suggest some replies! If you'd like to exit please type quit! ")

flag = True

while flag:

    user_input = raw_input('>>> ').lower() # get input and convert to lowercase

    if not re.search('quit', user_input):

        response = bot_response(user_input)

        for i in range(0, len(response)):
            print('* ' + response[i])

    else:

        flag = False
        # print("> Bye now!")
