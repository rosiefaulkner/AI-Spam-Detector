load_file_in_context("script.py")

try:
  if test_vectors != [tokens_to_bow_vector(test_doc, bow_sms_dictionary) for test_doc in test_spam_docs]:
    fail_tests("Is `test_vectors` defined as a list comprehension which calls `tokens_to_bow_vector()` on `test_doc` and `bow_sms_dictionary` for each `test_doc`?")
  
except NameError:
  fail_tests("Make sure to define `test_vectors`.")

pass_tests()