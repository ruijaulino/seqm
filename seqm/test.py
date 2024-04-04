


from abc import ABC, abstractmethod

class TemplateClass(ABC):
	@abstractmethod
	def required_method(self):
		"""Subclasses must implement this method"""
		pass

class ConcreteClass(TemplateClass):
	def required_method_2(self):
		print("This is the required method implemented in the subclass.")

# This will work
concrete_instance = ConcreteClass()
concrete_instance.required_method()

# Trying to instantiate the TemplateClass directly will raise an error
# template_instance = TemplateClass()

