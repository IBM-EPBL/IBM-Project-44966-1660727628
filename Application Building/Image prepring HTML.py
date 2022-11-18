<!DOCTYPE html>
	<html>
	<head>
	   <meta charset="UTF-8">
	    <meta name="viewport" content="width=device-width, initial-scale=1.0">
	    <meta http-equiv="X-UA-Compatible" content="ie=edge">
	    <title>Predict</title>
	    <link href="https://cdn.bootcss.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet">
	    <script src="https://cdn.bootcss.com/popper.js/1.12.9/umd/popper.min.js"></script>
	    <script src="https://cdn.bootcss.com/jquery/3.3.1/jquery.min.js"></script>
	    <script src="https://cdn.bootcss.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
	    <link href="{{ url_for('static', filename='css/main.css') }}" rel="stylesheet">   
	<style>
	body
	{
	    background-image: url("https://i.pinimg.com/originals/be/21/1a/be211ad5043a8d05757a3538bdd8f450.jpg");
	    background-size: cover;
	}
	.bar
	{
	margin: 0px;
	padding:20px;
	background-color:white;
	opacity:0.6;
	color:black;
	font-family:'Roboto',sans-serif;
	font-style: italic;
	border-radius:20px;
	font-size:25px;
	}
	a
	{
	color:grey;
	float:right;
	text-decoration:none;
	font-style:normal;
	padding-right:20px;
	}
	a:hover{
	background-color:black;
	color:white;
	border-radius:15px;
	font-size:30px;
	padding-left:10px;
	}
	.div1{
	  background-color: lightgrey;
	  width: 500px;
	  border: 10px solid peach;
	  padding: 20px;
	  margin: 20px;
	  height: 500px;
	}
	

	

	

	

	

	.header {	position: relative;
				top:0;	
				margin:0px;
				z-index: 1;
				left: 0px;
				right: 0px;
				position: fixed;
				background-color: #8B008B ;
				color: white;
				box-shadow: 0px 8px 4px grey;
				overflow: hidden;
				padding-left:20px;
				font-family: 'Josefin Sans';
				font-size: 2vw;
				width: 100%;
				height:8%;
				text-align: center;
			}
			.topnav {
	  overflow: hidden;
	  background-color: #FCAD98;
	}
	

	.topnav-right a {
	  float: left;
	  color: black;
	  text-align: center;
	  padding: 14px 16px;
	  text-decoration: none;
	  font-size: 18px;
	}
	

	.topnav-right a:hover {
	  background-color: #FF69B4;
	  color: black;
	}
	

	.topnav-right a.active {
	  background-color: #DA70D6;
	  color: black;
	}
	

	.topnav-right {
	  float: right;
	  padding-right:100px;
	}
	</style>
	</head>
	<body>
	<div class="header">    
	<div style="width:50%;float:left;font-size:2vw;text-align:left;color:black; padding-top:1%;padding-left:5%;">Nutrtion Image Analysis</div>
	  <div class="topnav-right"style="padding-top:0.5%;">
	    
	    <a href="{{ url_for('home')}}">Home</a>
	    <a  class="active" href="{{ url_for('image1')}}">Classify</a>
	  </div>
	</div>
	<br>
	

	

	</div>
	<div class="container">
	       <center>
	<div id="content" style="margin-top:2em">{% block content %}{% endblock %}</div></center>
	    </div>
	</body>
	

	<footer>
	    <script src="{{ url_for('static', filename='js/main.js') }}" type="text/javascript"></script>    
	</footer>
	

	</html>	

