namespace MauiAppPOC;

public partial class MainPage : ContentPage
{
	int count = 0;

	public MainPage()
	{
		InitializeComponent();
		
		// Display platform information
		PlatformLabel.Text = $"Running on: {DeviceInfo.Platform}\nDevice: {DeviceInfo.Model}\nVersion: {DeviceInfo.VersionString}";
	}

	private void OnCounterClicked(object sender, EventArgs e)
	{
		count++;

		if (count == 1)
			CounterBtn.Text = $"Clicked {count} time";
		else
			CounterBtn.Text = $"Clicked {count} times";

		SemanticScreenReader.Announce(CounterBtn.Text);
	}

	private void OnNameChanged(object sender, TextChangedEventArgs e)
	{
		if (string.IsNullOrWhiteSpace(e.NewTextValue))
		{
			GreetingLabel.Text = "Hello, Guest!";
		}
		else
		{
			GreetingLabel.Text = $"Hello, {e.NewTextValue}!";
		}
	}

	private async void OnNavigateClicked(object sender, EventArgs e)
	{
		var name = string.IsNullOrWhiteSpace(NameEntry.Text) ? "Guest" : NameEntry.Text;
		await Shell.Current.GoToAsync($"{nameof(DetailsPage)}?name={name}&count={count}");
	}
}
