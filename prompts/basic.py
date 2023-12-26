text = """
Solve the math problem. Think step-by-step. Always end your answer with 'The answer is <answer>.'
Q: {question}
"""

image = """
Solve the math problem using the image. Think step-by-step. Always end your answer with 'The answer is <answer>.'
Q: {question}
"""

TASK_PROMPT = {
'penguins_in_a_table':
                {'WITHOUT_IMAGE':'''Answer questions about a table of penguins and their attributes.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''Answer questions about a table of penguins and their attributes using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},     
'reasoning_about_colored_objects':
                {'WITHOUT_IMAGE':'''Answer extremely simple questions about the colors of objects on a surface.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''Answer extremely simple questions about the colors of objects on a surface using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'object_counting':
                {'WITHOUT_IMAGE':'''Questions that involve enumerating objects and asking the model to count them.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''Questions that involve enumerating objects and asking the model to count them using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'date_understanding':
                {'WITHOUT_IMAGE':'''Infer the date from context.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''Infer the date from context using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'geometric_shapes':
                {'WITHOUT_IMAGE':'''Name geometric shapes from their SVG paths.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''Name geometric shapes from their SVG paths and using the image .
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'navigate':
                {'WITHOUT_IMAGE':'''Given a series of navigation instructions, determine whether one would end up back at the starting point.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''Given a series of navigation instructions, determine whether one would end up back at the starting point using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'temporal_sequences':
                {'WITHOUT_IMAGE':'''Answer questions about which times certain events could have occurred.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''Answer questions about which times certain events could have occurred using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'tracking_shuffled_objects_three_objects':
                {'WITHOUT_IMAGE':'''A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'tracking_shuffled_objects_five_objects':
                {'WITHOUT_IMAGE':'''A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'tracking_shuffled_objects_seven_objects':
                {'WITHOUT_IMAGE':'''A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                ''',
                'WITH_IMAGE':'''A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps using the image.
            
                Q: {question}

                Please think step-by-step, and finally answer by selecting an option using the format "The answer is <option>"
                '''},
'maths':
               {'WITHOUT_IMAGE':'''Solve the math problem. Think step-by-step. Always end your answer with 'The answer is <answer>.

                Q: {question}''',
               'WITH_IMAGE':'''Solve the math problem using the image. Think step-by-step. Always end your answer with 'The answer is <answer>'.

                Q: {question}'''}
}
    
    
    
            

           
            
            
            
            
            