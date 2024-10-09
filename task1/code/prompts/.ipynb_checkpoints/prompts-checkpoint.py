argument_presence = f"""

        You are an AI assistant tasked with analyzing a comment about gay marriage in relation to a specific argument. You need to:
        - Identify if the comment makes use of the given argument. If it does, assign the label 1. If it does not, assign the label 0. 

        The argument to analyze is: {argument}
        
        Provide your response in the following JSON format:
        {{
            "comment": "full text of the comment",",
            "argument": "the argument being analyzed",
            "label": "the label for the use of the argument in the comment"
        }}

        Do NOT add additional text.
        
        Analyze the following comment in relation to the given argument:


"""