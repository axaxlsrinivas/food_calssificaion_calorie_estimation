namespace MauiAppPOC;

[QueryProperty(nameof(Name), "name")]
[QueryProperty(nameof(Count), "count")]
public partial class DetailsPage : ContentPage
{
	private string name = string.Empty;
	private int count;

	public string Name
	{
		get => name;
		set
		{
			name = value;
			NameLabel.Text = $"Name: {value}";
		}
	}

	public int Count
	{
		get => count;
		set
		{
			count = value;
			CountLabel.Text = $"Counter was at: {value}";
		}
	}

	public DetailsPage()
	{
		InitializeComponent();
	}

	private void OnCheckBatteryClicked(object sender, EventArgs e)
	{
		var battery = Battery.Default;
		var level = battery.ChargeLevel * 100;
		var state = battery.State;
		
		BatteryLabel.Text = $"Battery: {level:F0}% - {state}";
	}

	private void OnCheckNetworkClicked(object sender, EventArgs e)
	{
		var connectivity = Connectivity.Current;
		var access = connectivity.NetworkAccess;
		var profiles = connectivity.ConnectionProfiles;
		
		var profilesText = string.Join(", ", profiles);
		NetworkLabel.Text = $"Network: {access}\nProfiles: {profilesText}";
	}

	private async void OnGoBackClicked(object sender, EventArgs e)
	{
		await Shell.Current.GoToAsync("..");
	}
}
