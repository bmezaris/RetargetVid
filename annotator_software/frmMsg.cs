using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace VideoCropAnnotator
{
	public partial class Form2 : Form
	{
		public Form2()
		{
			InitializeComponent();
			this.DialogResult = DialogResult.Cancel;
		}

		private void button1_Click(object sender, EventArgs e)
		{
			this.DialogResult = DialogResult.Yes;
			this.Close();
		}

		private void button2_Click(object sender, EventArgs e)
		{
			this.DialogResult = DialogResult.No;
			this.Close();
		}

		private void Form2_Load(object sender, EventArgs e)
		{

		}
	}
}
