install: 
	@echo "Installing..."
	@pip install -r requirements.txt
	@echo "Done."

#test:
#	@echo "Testing using pytest..."
#	@pytest src/test.py
#	@echo "Done."

run:
	@echo "Running..."
	@python models/mlp.py
#	@python notebooks/face_detection.py
	@echo "Done."