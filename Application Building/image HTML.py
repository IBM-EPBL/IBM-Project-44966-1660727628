{% extends
 "imageprediction.html" %} {% block content %}
	<div style="float:left">
	<br>
	<br>
	<h5><font color="black" size="3" font-family="sans-serif"><b>Upload image to classify</b></font></h5><br><br>
	

	<div>
	    <form id="upload-file" method="post" enctype="multipart/form-data">
	       <label for="imageUpload" class="upload-label">
	            Choose...
	        </label>
	        <input type="file" name="file" id="imageUpload" accept=".png, .jpg, .jpeg">
	    </form>
	

	   <center> <div class="image-section" style="display:none;">
	        <div class="img-preview">
	            <div id="imagePreview">
	            </div></center>
	        </div>
	        <center><div>
	            <button type="button" class="btn btn-primary btn-lg " id="btn-predict">Classify</button>
	       </center></div>
	    </div>
	

	    <div class="loader" style="display:none;margin-left: 450px;"></div>
	

	    <h3 id="result">
	     
	        <span><p style="padding-top: 25px;"><h4>Food Classified is : <h4><b><u>{{showcase}}{{showcase1}}</p> </span>
	    </h3>
	

	</div>
	</div>
	

	       
	

	{% endblock %}

